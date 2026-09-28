import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data import Subset, Dataset
from tqdm import tqdm

from lavps.inference_models import get_inference_model
from lavps.inverse_problems import generate_inverse_problem
from lavps.networks import EpsilonNet
from lavps.paths import OUTPUT_DIRECTORY
from lavps.samplers.utils import (
    bridge_kernel_statistics, marginal_kernel_0_to_t_batch
)
from lavps.utils.checkpoint import CheckpointManager
from lavps.utils.experiment import clean_cache


@dataclass(frozen=True)
class Datapoint:
    """Structure containing the noisy state and context for an inverse problem step."""
    x0: torch.Tensor  # Original data point
    t: torch.Tensor  # Current noisy timestep (more noise)
    tau: torch.Tensor  # Target timestep (less noise)
    x0_hat: torch.Tensor  # Model's prediction of clean image from xt
    xt: torch.Tensor  # Noisy image at timestep t
    y: torch.Tensor  # Degraded measurement: H(x_orig) + noise


@dataclass
class TaskContext:
    """Container for inverse problem constants and models."""
    H_func: Any
    std: float
    model: EpsilonNet
    network_kwargs: dict[str, Any] # todo potentially put this into datapoint as it is data dependent


def update_h_func(H_func, seed_id, img_shape, deterministic, device):
    if hasattr(H_func, "update_with_seed"):
        H_func.update_with_seed(seed=seed_id if deterministic else None)
    elif hasattr(H_func, "update"):
        g = torch.Generator(device=device).manual_seed(seed_id) if deterministic else None
        H_func.update(imshape=img_shape, generator=g)


def loss_fn(
        vmean: torch.Tensor,
        vlog_std: torch.Tensor,
        data: Datapoint,
        ctx: TaskContext,
        mc_samples: int = 1,
) -> torch.Tensor:
    bs = data.y.shape[0]
    mean_prior, std_prior = bridge_kernel_statistics(
        data.xt, data.x0_hat, ctx.model, data.t, data.tau, 0, eta=1.0
    )
    std_prior = std_prior.reshape(std_prior.shape[0], 1, 1, 1)
    vmean_repeated = vmean.repeat(mc_samples, 1, 1, 1)
    x_tau = vmean_repeated + vlog_std.exp() * torch.randn_like(vmean_repeated)

    # Likelihood term
    x0_reconstruction = ctx.model.predict_x0(x_tau, data.tau, **ctx.network_kwargs)
    diff = data.y - ctx.H_func.H(x0_reconstruction)

    loss = (
            + 0.5 * torch.norm(diff) ** 2 / ctx.std ** 2 / mc_samples / bs
            + 0.5 * (((x_tau - mean_prior) / std_prior) ** 2).sum() / mc_samples / bs
            - vlog_std.sum() / bs
    )
    return loss


def adam_variational_approximation(
        data: Datapoint,
        ctx: TaskContext,
        lr: float,
        n_grad_steps: int,
        vmean: torch.Tensor | None = None,
        vlog_std: torch.Tensor | None = None,
):
    # Variational approximation: Initialization
    if vmean is None or vlog_std is None:
        mean_prior, std_prior = bridge_kernel_statistics(
            x_ell=data.xt, x_s=data.x0_hat, epsilon_net=ctx.model, ell=data.t, t=data.tau, s=0, eta=1.0
        )
        vmean = mean_prior.clone()
        vlog_std = std_prior.log() * torch.ones_like(data.xt)

    vmean = vmean.float().detach().requires_grad_()
    vlog_std = vlog_std.float().detach().requires_grad_()
    optim = torch.optim.Adam(params=[vmean, vlog_std], lr=lr)

    all_losses = []

    for _ in range(n_grad_steps + 1):  # +1 to include the initial loss check
        loss = loss_fn(vmean=vmean, vlog_std=vlog_std, data=data, ctx=ctx)
        all_losses.append(loss.item())

        if _ < n_grad_steps:
            optim.zero_grad()
            loss.backward()
            optim.step()

    return all_losses


def step_inference_model(
        *,
        data: Datapoint,
        ctx: TaskContext,
        inference_model: Any,
        compute_details: bool = False,
) -> dict[str, Any]:
    # 1. Get variational parameters from the inference model
    # We always need return_dict=True from the inner model to get prior params for normalization
    inference_results = inference_model(
        x0=data.x0_hat,
        xt=data.xt,
        y=data.y,
        H=ctx.H_func,
        timesteps_s=data.tau,
        timesteps_t=data.t,
        return_dict=True,
        **ctx.network_kwargs
    )

    vmean = inference_results["vmean"]
    vlog_std = inference_results["vlog_std"]

    # 2. Compute the primary loss
    loss_inference_model = loss_fn(vmean=vmean, vlog_std=vlog_std, data=data, ctx=ctx)

    # 3. Compute details
    if not compute_details:
        output = {"loss": loss_inference_model}
    else:
        with torch.no_grad():
            loss_prior = loss_fn(
                vmean=inference_results["mean_prior"],
                vlog_std=inference_results["log_std_prior"],
                data=data,
                ctx=ctx
            )
        output={
            "vmean": vmean,
            "vlog_std": vlog_std,
            "loss": loss_inference_model,
            "loss_inference_model": loss_inference_model.detach().item(),
            "loss_prior": loss_prior.item(),
            "ratio": (loss_inference_model / loss_prior).item()
        }
    return output


def new_datapoint(
        x_orig: torch.Tensor,
        ctx: TaskContext,
        low: int | None = None,
        high: int | None = None,
) -> Datapoint:
    timesteps = ctx.model.timesteps
    bs = x_orig.shape[0]

    low = low if low is not None else 1
    high = high if high is not None else len(timesteps) - 1

    idx = torch.randint(low, high, (bs,), dtype=torch.long, device=x_orig.device)
    tau = timesteps[idx]
    t = timesteps[idx + 1]

    # Measurement: y = H(x) + noise
    degraded = ctx.H_func.H(x_orig)
    y = degraded + ctx.std * torch.randn_like(degraded)

    # Latents: Sample xt and predict x0
    xt = marginal_kernel_0_to_t_batch(x_orig, t, ctx.model)
    x0_hat = ctx.model.predict_x0(xt, t, **ctx.network_kwargs)

    return Datapoint(x0=x_orig, t=t, tau=tau, x0_hat=x0_hat, xt=xt, y=y)


def get_train_val_datasets(cfg) -> tuple[Dataset, Dataset]:
    full_dataset_train = hydra.utils.instantiate(cfg.dataset)
    full_dataset_val = hydra.utils.instantiate(cfg.dataset)

    train_split = cfg.training.train_split
    val_split = cfg.training.val_split
    total_size = len(full_dataset_train)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)

    if cfg.training.shuffle:
        train_idx = list(range(train_size))
        val_idx = list(range(train_size, train_size + val_size))
    else:
        all_indices = torch.randperm(total_size).tolist()
        train_idx = all_indices[:train_size]
        val_idx = all_indices[train_size:train_size + val_size]
        print('Dataset shuffled')

    print(f'{val_size=}')
    print(f'{train_size=}')

    train_image_dataset = Subset(full_dataset_train, train_idx)
    val_image_dataset = Subset(full_dataset_val, val_idx)

    return train_image_dataset, val_image_dataset


def get_scheduler(optimizer, cfg, steps_per_epoch):
    if cfg.training.lr_scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    total_steps = cfg.training.num_epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * (0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(inference_model, H_func, epsilon_net, loader, optimizer, scheduler, cfg):
    inference_model.train()
    total_loss, n_samples = 0.0, 0
    pbar = tqdm(loader, leave=False, desc="Training")

    for seed_id, batch in enumerate(pbar):
        img = batch['image'].to(cfg.device, non_blocking=True)
        kw = {'class_labels': batch['label'].squeeze(1).to(cfg.device)} if 'label' in batch else {}

        # 1. Update degradation operator logic simplified
        update_h_func(H_func, seed_id, img.shape, cfg.task.deterministic, cfg.device)

        # 2. Setup Context and sample Bridge Datapoint
        ctx = TaskContext(H_func=H_func, std=cfg.inverse_problem.noise_level, model=epsilon_net, network_kwargs=kw)
        data = new_datapoint(x_orig=img, ctx=ctx, low=cfg.task.low_idx, high=cfg.task.high_idx)

        # 3. Forward pass & Loss (utilizing step_inference_model for consistency)
        optimizer.zero_grad()
        output = step_inference_model(
            data=data,
            ctx=ctx,
            inference_model=inference_model,
            compute_details=cfg.training.print_ratio
        )

        loss = output["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 4. Logging
        bs = img.shape[0]
        total_loss += loss.item() * bs
        n_samples += bs

        postfix = {"loss": f"{loss.item():.4f}"}
        if "ratio" in output:
            postfix["ratio"] = f"{output['ratio']:.3f}"
        pbar.set_postfix(postfix)

    return total_loss / n_samples


@hydra.main(config_path="../configs/", config_name="train_inference_model", version_base='1.2')
def main(cfg: DictConfig):

    clean_cache()
    torch.set_default_device(cfg.device)

    if cfg.get("output_path"):
        base_path = Path(cfg.output_path)
    else:
        base_path = OUTPUT_DIRECTORY
    save_path = base_path / Path(cfg.save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    torch.set_default_device(cfg.device)

    # Models & Data
    epsilon_net = hydra.utils.instantiate(cfg.model, device=cfg.device, num_steps=cfg.sampler.nsteps)
    train_ds, _ = get_train_val_datasets(cfg)
    generator = torch.Generator(device=cfg.device)
    train_loader = DataLoader(train_ds, cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, generator=generator)

    # H_func initialization
    res = train_ds[0]["image"].shape[-1]
    H_func = generate_inverse_problem(
        torch.randn(1, 3, res, res),
        cfg.inverse_problem.degradation,
        cfg.inverse_problem.noise_level
    ).H_func

    inference_model = get_inference_model(
        epsilon_net=epsilon_net, resolution=res, device=cfg.device,
        inference_model_type=cfg.inference_model_type
    ).to(cfg.device)

    # Optimization Setup
    optimizer = torch.optim.AdamW(inference_model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.wd)
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))

    # Checkpoint Manager
    ckpt_mgr = CheckpointManager(
        save_path,
        num_to_keep=cfg.checkpoint.num_newest_to_keep,
        keep_list=cfg.checkpoint.get('keep_list', [])
    )

    # Training Loop
    for epoch in range(1, cfg.training.num_epochs + 1):
        avg_loss = train_one_epoch(inference_model, H_func, epsilon_net, train_loader, optimizer, scheduler, cfg)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")
        ckpt_mgr.save(inference_model, optimizer, epoch, avg_loss)


if __name__ == "__main__":
    main()