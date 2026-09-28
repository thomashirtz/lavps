from typing import Tuple, Callable

import torch
from torch import Tensor
from tqdm import tqdm

from lavps.inverse_problems import InverseProblem
from lavps.networks import EpsilonNet
from lavps.samplers.ddim import ddim_step
from lavps.samplers.utils import (
    bridge_kernel_statistics,
    sample_bridge_kernel,
)


def mgdm(
        initial_noise: torch.Tensor,
        inverse_problem: InverseProblem,
        epsilon_net: EpsilonNet,
        n_reps: int,
        lr_fn: float,
        initialize_va_fn: bool,
        gradient_steps_fn: int,
        n_denoising_steps: int,
        eta: float = 1.0,
        tau_sampling: str = "mix",
        threshold: int = 70,
        min_tau: int = 10,
        initialize_va_cfg=None,
        model_for_initialization=None,
        network_kwargs=None,
) -> Tensor:
    timesteps = epsilon_net.timesteps
    log_pot = inverse_problem.log_pot
    n_steps = len(timesteps)

    network_kwargs = network_kwargs or {}

    gradsteps_fn = get_gradientsteps_fn(gradient_steps_fn, n_steps)
    lr_fn = get_lr_fn(lr_fn, n_steps)
    tau_sample_fn = get_tau_sampling(
        type=tau_sampling, min_tau=min_tau, threshold=threshold
    )

    initialize_va_fn = get_inference_model_fn(initialize_va_fn, n_steps)

    x_t = initial_noise
    x_0 = epsilon_net.predict_x0(x_t, epsilon_net.timesteps[-1], **network_kwargs)

    flipd_timesteps = timesteps.flip(0)
    pbar = tqdm(
        enumerate(
            zip(flipd_timesteps[2:-1], flipd_timesteps[1:-2], flipd_timesteps[:-2])
        )
    )

    for idx, (t_prev, t, t_next) in pbar:

        tau = tau_sample_fn(t_prev, idx)
        ngrad_steps = gradsteps_fn(idx)
        lr = lr_fn(idx)

        x_tnext = x_t
        with torch.no_grad():
            x_t = sample_bridge_kernel(x_tnext, x_0, epsilon_net, t_next, t, 0, eta=eta)

        # log_pot_fn = lambda x: log_pot(epsilon_net.predict_x0(x, tau, **network_kwargs))
        def log_pot_fn(x, kwargs=None): # TODO maybe make this code better but I do not have good idea yet how to improve that
            if kwargs is None:
                kwargs = network_kwargs
            return log_pot(epsilon_net.predict_x0(x, tau, **kwargs))

        for _ in range(n_reps):
            x_0, x_t = gibbs_step(
                # fmt: off
                x_0, x_t, x_tnext,
                tau, t, t_next,
                # fmt: on
                epsilon_net,
                log_pot_fn,
                lr,
                ngrad_steps,
                n_denoising_steps,
                eta=eta,
                inv_prob=inverse_problem,
                model_for_initialization=model_for_initialization,
                initialization_type=initialize_va_cfg.type,
                initialize_with_model=initialize_va_fn(idx),
                compare_loss=initialize_va_cfg.compare_loss,
                network_kwargs=network_kwargs,
                initialization_best_of=initialize_va_cfg.initialization_best_of,
            )

    return x_0


def gibbs_step(
        # fmt: off
        x_0, x_t, x_tnext,
        tau, t, t_next,
        # fmt: on
        epsilon_net: EpsilonNet,
        log_pot: Callable[[Tensor], Tensor],
        lr: float,
        n_gradient_steps: int,
        n_denoising_steps: int,
        eta: float,
        inv_prob: InverseProblem,
        model_for_initialization,
        initialization_type,
        initialize_with_model,
        compare_loss,
        network_kwargs=None,
        initialization_best_of: bool = False,
) -> Tuple[Tensor, Tensor]:
    network_kwargs = network_kwargs or {}

    # potential
    x_tau = _pot_update_stochastic(
        # fmt: off
        x_0, x_t,
        tau, t,
        # fmt: on
        log_pot,
        epsilon_net,
        lr,
        n_gradient_steps,
        inv_prob=inv_prob,
        model_for_initialization=model_for_initialization,
        initialization_type=initialization_type,
        initialize_with_model=initialize_with_model,
        compare_loss=compare_loss,
        initialization_best_of=initialization_best_of,
        network_kwargs=network_kwargs,
    )

    # denoising step
    with torch.no_grad():
        x_0 = _denoising_step(0, tau, x_tau, epsilon_net, n_denoising_steps, eta=eta, **network_kwargs)

    # noising step
    with torch.no_grad():
        x_t = sample_bridge_kernel(x_tnext, x_tau, epsilon_net, t_next, t, tau)

    return x_0, x_t


def compute_kl_loss(vmean, vlog_std, mean_prior, std_prior, log_pot, mc_samples, reduce=True, network_kwargs=None):
    kwargs = {} if network_kwargs is None else network_kwargs
    # Ensure vmean is 5D: [num_candidates, batch_size, C, H, W]
    if vmean.ndim == 4:
        vmean = vmean.unsqueeze(0)
        vlog_std = vlog_std.unsqueeze(0)

    num_cands, batch_size = vmean.shape[0], vmean.shape[1]

    # Flatten candidates and batch into one dimension: [num_cands * batch_size, C, H, W]
    vmean_flat = vmean.flatten(0, 1)
    vlog_std_flat = vlog_std.flatten(0, 1)

    # Apply MC samples: [num_cands * batch_size * mc_samples, C, H, W]
    vmean_mc = vmean_flat.repeat_interleave(mc_samples, dim=0)
    vlog_std_mc = vlog_std_flat.repeat_interleave(mc_samples, dim=0)
    vsample = vmean_mc + vlog_std_mc.exp() * torch.randn_like(vmean_mc)

    # Single forward pass: returns [num_cands * batch_size * mc_samples]
    if kwargs: # TODO basically need to double any conditioning, for now we only do class_labels
        kwargs = {'class_labels': network_kwargs['class_labels'].repeat_interleave(2, 0)}
        lp = log_pot(vsample, kwargs=kwargs)
    else:
        lp = log_pot(vsample)

    # 1. Potential term: mean over MC, sum over Batch, result is [num_cands]
    lp_term = -lp.view(num_cands, batch_size, mc_samples).mean(dim=2).sum(dim=1)

    # 2. KL Distance term
    # vsample_5d: [num_cands, batch_size, mc_samples, C, H, W]
    # mean_prior: [batch_size, C, H, W]
    vsample_5d = vsample.view(num_cands, batch_size, mc_samples, *vmean.shape[-3:])
    mean_prior_broadcast = mean_prior.view(1, batch_size, 1, *vmean.shape[-3:])

    kl_dist = 0.5 * (((vsample_5d - mean_prior_broadcast) / std_prior) ** 2)
    kl_dist = kl_dist.view(num_cands, -1).sum(dim=1) / mc_samples

    # 3. Entropy term
    ent_term = -vlog_std.view(num_cands, -1).sum(dim=1)

    loss = lp_term + kl_dist + ent_term

    if False: # this branch is just to compare with the old implementation, will be deleted as soon as the new implementation is accepted
        # _vmean_repeated = vmean.repeat(mc_samples, 1, 1, 1)
        # _vsample = _vmean_repeated + vlog_std.exp() * torch.randn_like(vmean_repeated)
        _vsample = vsample

        a = -log_pot(_vsample) / mc_samples
        b = 0.5 * (((_vsample - mean_prior) / std_prior) ** 2).sum() / mc_samples
        c = vlog_std.sum()
        _loss = (
                a
                + b
                - c
        )

    return loss.sum() if reduce else loss


def compute_exact_kl_loss(vmean, vlog_std, mean_prior, std_prior, log_pot, mc_samples=1, beta=1.0, ):
    """
    Args
    ----
    vmean:       tensor like (B, ...) — posterior mean
    vlog_std:    tensor like (B, ...) — posterior log-std (natural log)
    mean_prior:  tensor like (B, ...) or broadcastable to vmean
    std_prior:   0-D tensor (scalar) or broadcastable — prior std
    log_pot:     callable: takes a batch of samples shaped like vmean and returns per-example log-prob tensor (B,)
    mc_samples:  int — number of MC samples for the data term

    Returns
    -------
    loss:        scalar — mean over batch of [ MC NLL + exact KL(q||p) ]
    nll:         tensor (B,) — per-example Monte Carlo NLL
    kl:          tensor (B,) — per-example exact KL
    """
    B = vmean.shape[0]
    vstd = vlog_std.exp()
    vvar = vstd.pow(2)

    # --- Monte Carlo NLL (data term) ---
    vmean_repeated = vmean.repeat(mc_samples, 1, 1, 1)
    vsample = vmean_repeated + vlog_std.exp() * torch.randn_like(vmean_repeated)
    nll_mean = -log_pot(vsample) / mc_samples

    # --- Closed-form KL(q||p) for diagonal/isotropic prior ---
    prior_var = std_prior.pow(2)
    # Elementwise KL, then sum over non-batch dims
    kl_elem = 0.5 * (
            (vvar + (vmean - mean_prior).pow(2)) / prior_var
            - 1.0
            + 2.0 * (std_prior.log() - vlog_std)  # = log(prior_var) - log(post_var)
    )
    kl = kl_elem.view(B, -1).sum(dim=1)  # (B,)
    kl_mean = kl.mean()

    loss = nll_mean + beta * kl_mean
    return loss, nll_mean, beta * kl_mean


# ------ utils
def _pot_update_stochastic(
        # fmt: off
        x_0, x_t,
        tau, t,
        # fmt: on
        log_pot: Callable[[Tensor], Tensor],
        epsilon_net: EpsilonNet,
        lr: float,
        n_gradient_steps: int,
        mc_samples: int = 1,  # Number of Monte Carlo samples to compute KL: the loss becomes more stable (less spikes)
        plot_loss: bool = False,
        save_context: bool = False,
        inv_prob: InverseProblem = None,
        model_for_initialization=None,
        initialization_type=None,
        initialize_with_model=False,
        compare_loss=False,
        network_kwargs=None,
        initialization_best_of: bool = False,
) -> Tensor:
    """
    Return
    ------
    x_tau
    """

    network_kwargs = network_kwargs or {}

    with torch.no_grad():
        mean_prior, std_prior = bridge_kernel_statistics(
            x_t, x_0, epsilon_net, t, tau, 0, eta=1.0
        )
        log_std_prior = std_prior.log()

    if not initialize_with_model:
        # zero-shot initialization
        vmean = mean_prior.clone()
        vlog_std = log_std_prior * torch.ones_like(x_t)
    else:
        if initialization_type == "inference_model":
            # inference model initialization
            with torch.no_grad():
                vmean, vlog_std = model_for_initialization(x_0, x_t, inv_prob.obs, inv_prob.H_func, tau.view(-1),
                                                           t.view(-1), **network_kwargs)
        elif initialization_type == "conditional_denoiser":
            with torch.no_grad():
                x_0_conditional = model_for_initialization.predict_x0(x_t, t, H=inv_prob.H_func, y=inv_prob.obs)
                mean_prior_conditional, std_prior_conditional = bridge_kernel_statistics(
                    x_t, x_0_conditional, model_for_initialization, t, tau, 0, eta=1.0
                )
                log_std_prior_conditional = std_prior_conditional.log()
                vmean = mean_prior_conditional.clone()
                vlog_std = log_std_prior_conditional * torch.ones_like(x_t)
        else:
            raise NotImplementedError

    if initialize_with_model and initialization_best_of:
        # 1. Create two independent leaf candidate (to avoid error "ValueError: can't optimize a non-leaf Tensor")
        # Candidate 0: From Model
        vmean_0 = vmean.detach().clone().requires_grad_(True)
        vlog_std_0 = vlog_std.detach().clone().requires_grad_(True)

        # Candidate 1: Zero-shot (mean_prior)
        vmean_1 = mean_prior.detach().clone().requires_grad_(True)
        vlog_std_1 = log_std_prior.expand_as(vlog_std).detach().clone().requires_grad_(True)

        # 2. Stack them just for the comparison (this is a non-leaf operation)
        vmean_stack = torch.stack([vmean_0, vmean_1])
        vlog_std_stack = torch.stack([vlog_std_0, vlog_std_1])

        # 3. Compute losses (DO NOT use torch.no_grad() here if you want to reuse initial_loss)
        losses = compute_kl_loss(vmean_stack, vlog_std_stack, mean_prior, std_prior, log_pot, mc_samples, reduce=False, network_kwargs=network_kwargs)

        # 4. Identify the winner
        with torch.no_grad():
            best_idx = torch.argmin(losses).item()

        # 5. Point vmean/vlog_std to the original LEAF tensors
        if best_idx == 0:
            vmean, vlog_std = vmean_0, vlog_std_0
        else:
            vmean, vlog_std = vmean_1, vlog_std_1

        # 6. Keep the specific loss associated with the winner
        initial_loss = losses[best_idx]
    else:
        vmean = vmean.detach().clone().requires_grad_(True)
        vlog_std = vlog_std.detach().clone().requires_grad_(True)
        initial_loss = None

    optim = torch.optim.Adam(params=[vmean, vlog_std], lr=lr)

    all_losses = []  # <-- keeps every loss value in the order produced
    step_counter = 0  # optional, for x‑axis labelling

    for i in range(n_gradient_steps):
        optim.zero_grad()

        if i == 0 and initial_loss is not None:
            loss = initial_loss
        else:
            loss = compute_kl_loss(vmean, vlog_std, mean_prior, std_prior, log_pot, mc_samples)

        loss.backward()
        optim.step()

        all_losses.append(loss.item())
        step_counter += 1

    with torch.no_grad():
        x_tau = vmean + vlog_std.exp() * torch.randn_like(x_t)

    return x_tau


def _denoising_step(
        s, tau, x_tau: Tensor, epsilon_net: EpsilonNet, n_denoising_steps: int, eta: float, **kwargs,
) -> Tensor:
    if isinstance(s, int):
        s = torch.tensor(s, dtype=torch.int32)
    if isinstance(tau, int):
        tau = torch.tensor(tau, dtype=torch.int32)

    s = s.clamp(0, None)
    n_steps = min(n_denoising_steps, tau - s + 1)

    if n_steps < 2:
        return epsilon_net.predict_x0(x_tau, tau, **kwargs)

    arr_times = torch.linspace(s, tau, n_steps, dtype=torch.int32).flip(0)
    sequence_k_k_prev = zip(arr_times[:-1], arr_times[1:])

    x_k = x_tau
    for k, k_prev in sequence_k_k_prev:
        x_k = ddim_step(x_k, epsilon_net, k, k_prev, eta=eta, **kwargs)

    return x_k


def get_tau_sampling(type="uniform", **kwargs):
    if type == "uniform":
        min_tau = kwargs.get("min_tau")
        return lambda t, idx: torch.randint(min_tau, t, (1,))[0]
    elif type == "midpoint":
        return lambda t, idx: t // 2
    elif type == "zero":
        return lambda t, idx: torch.randint(1, max(1, t // 5), (1,))[0]
    elif type == "mix":
        threshold = kwargs.get("threshold")
        min_tau = kwargs.get("min_tau")
        return lambda t, idx: (
            torch.randint(min_tau, int(t), (1,))[0] if idx < threshold else t
        )
    elif type == "deterministic":
        return lambda t, idx: t
    elif type == "mix-det":
        threshold = kwargs.get("threshold")
        return lambda t, idx: (75 * t) // 100 if idx < threshold else t
    else:
        raise ValueError("time sampler not implemented")


def get_gradientsteps_fn(gradientsteps_fn, n_steps):
    def fn(i):
        for rule in gradientsteps_fn.conditions:
            condition = rule.condition
            if condition == "default" or eval(condition, {"i": i, "n_steps": n_steps}):
                return rule["return"]

    return fn


def get_inference_model_fn(inference_model_fn, n_steps):
    def fn(i):
        for rule in inference_model_fn.conditions:
            condition = rule.condition
            if condition == "default" or eval(condition, {"i": i, "n_steps": n_steps}):
                return rule["return"]

    return fn


def get_lr_fn(lr_fn, n_steps):
    def fn(i):
        for rule in lr_fn.conditions:
            condition = rule.condition
            if condition == "default" or eval(condition, {"i": i, "n_steps": n_steps}):
                return rule["return"]

    return fn


def get_beta_fn(beta_fn, n_steps):
    def fn(i):
        for rule in beta_fn.conditions:
            condition = rule.condition
            if condition == "default" or eval(condition, {"i": i, "n_steps": n_steps}):
                return rule["return"]

    return fn
