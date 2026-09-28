from dataclasses import dataclass
from typing import Callable, Any

import torch

from invfussion.operators.base import H_functions as H_functions_invfussion
from invfussion.operators.painting import MissingBoxCustom, CenterBox, MissingSmallPatches
from invfussion.operators.super_resolution import SuperResolution
from lavps.operators.base import H_functions as H_functions_lavps
from lavps.operators.blur import Blur, MotionBlur
from lavps.operators.identity import Identity

H_functions = H_functions_lavps | H_functions_invfussion | Any


@dataclass
class InverseProblem:
    ref: torch.Tensor = None
    obs: torch.Tensor = None
    H_func: H_functions = None
    std: float = None
    log_pot: Callable[[torch.Tensor], torch.Tensor] = None
    obs_image: torch.Tensor = None


def get_degradation_from_task(
        task: str,
        img_shape: torch.Size,
        device: torch.device,
        seed: int = None
) -> H_functions:
    """Parses a task string and returns the corresponding Degradation object."""
    B, C, H, W = img_shape
    parts = task.split("-")
    name = parts[0]

    # Check for stochastic prefix
    is_stochastic = name.startswith("stochastic_")
    # Determine actual task key (e.g., 'stochastic_invfussion_box' -> 'invfussion_box')
    clean_name = name.replace("stochastic_", "")

    # 1. Linear Operators / Simple Tasks
    if clean_name == "denoising":
        # Format: denoising
        return Identity()

    elif clean_name == "super_resolution":
        # Format: super_resolution-{channels}-{img_dim}-{ratio}
        channels, img_dim, ratio = int(parts[1]), int(parts[2]), int(parts[3])
        return SuperResolution(
            imshape=(1, channels, img_dim, img_dim),
            ratio=ratio,
            device=device,
        )

    elif clean_name == "gaussian_blur":
        # Format: gaussian_blur-{kernel_size}-{intensity}
        return Blur(
            kernel_size=int(parts[1]),
            intensity=int(parts[2]),
            device=device,
        )

    elif clean_name == "motion_blur":
        # Format: [stochastic_]motion_blur-{kernel_size}-{intensity}-{img_dim}
        kernel_size, intensity, img_dim = int(parts[1]), float(parts[2]), int(parts[3])
        m_seed = seed if is_stochastic else 0
        return MotionBlur(
            kernel_size=kernel_size,
            intensity=intensity,
            device=device,
            seed=m_seed,
            imshape=(1, C, img_dim, img_dim),
        )

    elif clean_name == "invfussion_box":
        # Format: [stochastic_]invfussion_box-{channels}-{img_dim}-{min_frac}-{max_frac}
        m_seed = seed if is_stochastic else 0
        gen = torch.Generator(device=device).manual_seed(m_seed) if m_seed is not None else None
        channels, img_dim, min_f, max_f = int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])
        return MissingBoxCustom(
            imshape=(1, channels, img_dim, img_dim),
            device=device,
            generator=gen,
            min_frac=min_f,
            max_frac=max_f,
        )

    elif clean_name == "invfussion_smallpatch":
        # Format: [stochastic_]invfussion_smallpatch-{channels}-{img_dim}-{min_p}-{max_p}
        m_seed = seed if is_stochastic else 0
        gen, gen_cpu = None, None
        if m_seed is not None:
            gen = torch.Generator(device=device).manual_seed(m_seed)
            gen_cpu = torch.Generator(device=device).manual_seed(m_seed)
        channels, img_dim, min_p, max_p = int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])
        return MissingSmallPatches(
            imshape=(1, channels, img_dim, img_dim),
            min_p=min_p,
            max_p=max_p,
            device=device,
            generator=gen,
            gen_cpu=gen_cpu,
        )

    elif clean_name == "invfussion_center_box":
        # Format: invfussion_center_box-{channels}-{img_dim}-{factor}
        channels, img_dim, factor = int(parts[1]), int(parts[2]), int(parts[3])
        return CenterBox(
            imshape=(1, channels, img_dim, img_dim),
            factor=factor,
            device=device,
        )

    raise ValueError(f"Unknown degradation task: {task}")

def generate_inverse_problem(
        images: torch.Tensor,
        degradation: H_functions | str,
        noise_level: float,
        seed: int = None,
) -> InverseProblem:
    """
    Unified entry point. Handles images: [B, C, H, W] OR [C, H, W].
    """
    device = images.device
    dtype = images.dtype

    # Force 4D batch shape for consistent operator behavior
    is_single_image = images.ndim == 3
    if is_single_image:
        images = images.unsqueeze(0)  # [1, C, H, W]

    # Resolve operator
    if isinstance(degradation, str):
        H_func = get_degradation_from_task(degradation, images.shape, device, seed=seed)
    else:
        H_func = degradation

    # y = H(x) + n
    y = H_func.H(images)
    y = y + noise_level * torch.randn(y.shape, device=device, dtype=dtype)

    # Generate visual observation proxy
    obs_image = H_func.H_pinv(y).reshape(images.shape)
    obs_image = obs_image.clamp(-1.0, 1.0) # Fixed: assigning back

    def log_potential(x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        diff = y - H_func.H(x)
        diff_flat = diff.reshape(x.shape[0], -1)
        return -0.5 * torch.sum(diff_flat ** 2, dim=1) / (noise_level ** 2)

    # Squeeze results back if input was 3D
    ref_out = images.squeeze(0) if is_single_image else images
    obs_image_out = obs_image.squeeze(0).cpu() if is_single_image else obs_image.cpu()

    return InverseProblem(
        ref=ref_out,
        obs=y,
        H_func=H_func,
        log_pot=log_potential,
        std=noise_level,
        obs_image=obs_image_out,
    )
