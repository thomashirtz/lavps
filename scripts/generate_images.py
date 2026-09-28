import os
os.environ['HYDRA_FULL_ERROR'] = '1'

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from lavps.inverse_problems import generate_inverse_problem
from lavps.utils.experiment import get_torch_dtype, get_true_device_id, fix_seed, save_im, seed_from_name
from lavps.utils.metrics import LPIPS, PSNR, SSIM
from lavps.utils.profiling import ExecutionProfiling
from lavps.paths import OUTPUT_DIRECTORY
from lavps.samplers.dps import dps
from lavps.samplers.mgdm import mgdm
from lavps.inference_models import get_inference_model

torch.cuda.empty_cache()
fix_seed(seed=0)


@hydra.main(config_path="../configs/", config_name="generate_images", version_base='1.2')
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    yaml_str = OmegaConf.to_yaml(cfg)
    print('\n' + yaml_str)

    # Device handling
    device = torch.device(cfg.device)
    device_true_id = get_true_device_id(device_id=device.index)
    print(f"Using {device.index=}, {device_true_id=}")

    # Dtype handling
    dtype = get_torch_dtype(cfg.dtype) # todo why isn't used ?

    experiment_folder = cfg.experiment.folder
    output_directory = OUTPUT_DIRECTORY
    if experiment_folder is not None:
        output_directory = OUTPUT_DIRECTORY / str(experiment_folder)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Handling the output files and folders
    experiment_name = cfg.experiment.name + f'-device_id={device_true_id}'
    config_file_path = output_directory / f"config-{experiment_name}.yaml"
    config_file_path.write_text(yaml_str, encoding='utf-8')
    results_file_path = output_directory / f'results-{experiment_name}.csv'

    image_folder = output_directory / (cfg.experiment.name + f'-device_id={device_true_id}')
    image_folder.mkdir(exist_ok=True, parents=True)

    sampler = {
        "lavps": mgdm,
        "mgdm": mgdm,
        "dps": dps,
    }[cfg.sampler.name]

    # Device Configuration
    if cfg.failsafe_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Failsafe.')
    print(f"Cuda is available")
    torch.set_default_device(device)

    dataset = hydra.utils.instantiate(cfg.dataset)

    num_steps = cfg.sampler.nsteps
    batch_size = 1

    first_write = True
    dataloader = DataLoader(dataset, batch_size=batch_size)
    epsilon_net = hydra.utils.instantiate(cfg.model, device=device, num_steps=num_steps)

    # Is it a latent diffusion model?

    # Model initialization only for amortized MGDM (do it once for all)
    if cfg.sampler.name in ["mgdm", "lavps"] and cfg.sampler.parameters.initialize_va_cfg.get('type', '') == 'inference_model':
        model_for_initialization = get_inference_model(
            epsilon_net=epsilon_net,
            inference_model_type=cfg.sampler.parameters.initialize_va_cfg.model,
            path=cfg.sampler.parameters.initialize_va_cfg.path,
            device=device,
            set_eval_mode=True,
            resolution=cfg.model.image_resolution,
        )
    else:
        model_for_initialization = None


    for batch_id, batch in enumerate(dataloader):
        batch_image = batch["image"].to(device)
        batch_name = batch["name"]
        true_batch_size = batch_image.shape[0]

        if cfg.inverse_problem.degradation.startswith("stochastic"):
            assert len(batch_name) == 1
            name = batch_name[0]
            assert isinstance(name, str)
            operator_seed = seed_from_name(name, salt=str(cfg.inverse_problem.degradation))
        else:
            operator_seed = batch_id

        inverse_problem = generate_inverse_problem(
            images=batch_image,
            degradation=cfg.inverse_problem.degradation,
            noise_level=cfg.inverse_problem.noise_level,
            seed=operator_seed,
        )
        initial_noise = torch.randn_like(inverse_problem.ref, device=inverse_problem.ref.device)

        kwargs = {}
        if batch.get('label', None) is not None:
            kwargs = dict(network_kwargs={'class_labels': batch.get('label').squeeze(axis=1).to(device)})

        with ExecutionProfiling(num_iterations=num_steps, batch_size=true_batch_size) as inference_stats:
            reconstructions = sampler(
                initial_noise=initial_noise,
                inverse_problem=inverse_problem,
                epsilon_net=epsilon_net,
                model_for_initialization=model_for_initialization,
                **cfg.sampler.parameters,
                **kwargs,
            )

        reconstructions = reconstructions.clamp(min=-1, max=1)
        observations = inverse_problem.obs_image.clamp(min=-1, max=1)
        references = inverse_problem.ref.clamp(min=-1, max=1)

        lpips, ssim, psnr = LPIPS(), SSIM(), PSNR()

        labels = {'labels': batch.get('label_digit')} if batch.get('label_digit') is not None else {}
        metrics = {
            'lpips': lpips.score(reconstructions, references).reshape(true_batch_size),
            'ssim': ssim.score(reconstructions, references).reshape(true_batch_size),
            'psnr': psnr.score(reconstructions, references).reshape(true_batch_size),
            **labels,
        }

        print(f"{cfg.sampler} metrics")
        print(f"{'lpips':10}: {metrics['lpips'][0]}")
        print(f"{'ssim':10}: {metrics['ssim'][0]}")
        print(f"{'psnr':10}: {metrics['psnr'][0]}")
        print("===================")
        print(f"{'runtime':10}: {inference_stats['time_per_image']}")
        print(f"{'GPU':10}: {inference_stats['max_memory_reserved_during_sampling']}")

        for image_id in range(true_batch_size):
            save_im(
                batch_image[image_id].cpu(),
                save_path=image_folder / f"{batch_name[image_id]}-reference.png",
            )
            save_im(
                observations[image_id].detach().cpu(),
                save_path=image_folder / f"{batch_name[image_id]}-observation.png",
            )
            save_im(
                reconstructions[image_id].detach().cpu(),
                save_path=image_folder / f"{batch_name[image_id]}-reconstruction.png",
            )
            print(image_folder / f"{batch_name[image_id]}-{{reconstruction/reference/observation}}.png")
            print(f'{batch_id=}') # Print which batch we just processed, useful when dealing with ImageNet
        lst = [
            {
                'image_name': n,
                'seed': operator_seed,
                **{k: float(v[i]) for k, v in metrics.items()},
                **inference_stats,
            }
            for i, n in enumerate(batch_name)
        ]
        tmp_df = pd.DataFrame(lst)
        tmp_df.to_csv(results_file_path,
                      mode='w' if first_write else 'a',
                      header=first_write,
                      index=False)
        first_write = False


if __name__ == '__main__':
    main()
