import hydra
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from lavps.inverse_problems import generate_inverse_problem
from lavps.paths import OUTPUT_DIRECTORY
from lavps.utils.experiment import fix_seed, save_im, seed_from_name

torch.cuda.empty_cache()
fix_seed(seed=0)


@hydra.main(config_path="../configs/", config_name="generate_images", version_base='1.2')
def main(cfg: DictConfig):
    yaml_str = OmegaConf.to_yaml(cfg)
    print('\n' + yaml_str)
    save_im_path = OUTPUT_DIRECTORY / 'generate_observations'
    save_im_path.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    torch.set_default_device(device)


    # Device Configuration
    if cfg.failsafe_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Failsafe.')
    print(f"Cuda is available")
    torch.set_default_device(device)

    dataset = hydra.utils.instantiate(cfg.dataset)

    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for batch_id, batch in enumerate(dataloader):
        batch_image = batch["image"].to(device)
        batch_name = batch["name"][0]

        if cfg.inverse_problem.degradation.startswith("stochastic"):
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

        print(save_im_path / f"{batch_name}_observation/reference.png")
        save_im(
            inverse_problem.obs_image.detach().cpu(),
            save_path=save_im_path / f"{batch_name}_observation.png",
            title="Observation",
        )
        save_im(
            inverse_problem.ref.cpu(),
            save_path=save_im_path / f"{batch_name}_reference.png",
            title="Reference",
        )


if __name__ == "__main__":
    main()
