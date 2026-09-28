from typing import Optional
from typing import Tuple

import torch

from lavps.networks import EpsilonNet


def marginal_kernel_0_to_t_batch(x0: torch.Tensor, t: torch.Tensor, epsilon_net):
    assert t.ndim == 1 and t.shape[0] == x0.shape[0], "Shapes: x0 (B,…) and t (B,) must match"

    device, dtype = x0.device, x0.dtype

    # ᾱ_t : (B,)  ← gather cumulative alphas for each sample’s t_i
    alpha_bar_t = epsilon_net.acp_f8[t].to(device=device).to(dtype=dtype)

    # reshape to broadcast over (C,H,W)
    alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)  # (B,1,1,1)
    sqrt_alpha_t = alpha_bar_t.sqrt()  # √ᾱ_t
    std_t = (1.0 - alpha_bar_t).sqrt()  # √(1−ᾱ_t)

    mean_t = sqrt_alpha_t * x0  # μ_t
    eps = torch.randn_like(x0)  # ε ~ 𝒩(0,I)
    x_t = mean_t + std_t * eps  # sample

    return x_t


def bridge_kernel_statistics(
        x_ell: torch.Tensor,
        x_s: torch.Tensor,
        epsilon_net: EpsilonNet,
        ell: int,
        t: int,
        s: int,
        eta: float = 1.0,
):
    """s < t < ell"""
    f8 = torch.float64

    alpha_cum_s_to_t = epsilon_net.acp_f8[t] / epsilon_net.acp_f8[s]
    alpha_cum_t_to_ell = epsilon_net.acp_f8[ell] / epsilon_net.acp_f8[t]
    alpha_cum_s_to_ell = epsilon_net.acp_f8[ell] / epsilon_net.acp_f8[s]
    std = (
            eta
            * ((1 - alpha_cum_t_to_ell) * (1 - alpha_cum_s_to_t) / (1 - alpha_cum_s_to_ell))
            ** 0.5
    )
    coeff_xell = ((1 - alpha_cum_s_to_t - std ** 2) / (1 - alpha_cum_s_to_ell)) ** 0.5
    coeff_xs = (alpha_cum_s_to_t ** 0.5) - coeff_xell * (alpha_cum_s_to_ell ** 0.5)

    coeff_xell, coeff_xs, std = (
        coeff_xell.to(dtype=f8),
        coeff_xs.to(dtype=f8),
        std.to(dtype=f8),
    )
    coeff_xell = coeff_xell.reshape(-1, 1, 1, 1)
    coeff_xs = coeff_xs.reshape(-1, 1, 1, 1)
    return (coeff_xell * x_ell + coeff_xs * x_s).float(), std.float()


def bridge_kernel_all_stats(
        ell: int,
        t: int,
        s: int,
        epsilon_net: EpsilonNet,
        eta: float = 1.0,
) -> Tuple[float, float, float]:
    """s < t < ell

    Return
    ------
    coeff_xell, coeff_xs, std
    """
    f8 = torch.float64

    alpha_cum_s_to_t = epsilon_net.acp_f8[t] / epsilon_net.acp_f8[s]
    alpha_cum_t_to_ell = epsilon_net.acp_f8[ell] / epsilon_net.acp_f8[t]
    alpha_cum_s_to_ell = epsilon_net.acp_f8[ell] / epsilon_net.acp_f8[s]
    std = (
            eta
            * ((1 - alpha_cum_t_to_ell) * (1 - alpha_cum_s_to_t) / (1 - alpha_cum_s_to_ell))
            ** 0.5
    )
    coeff_xell = ((1 - alpha_cum_s_to_t - std ** 2) / (1 - alpha_cum_s_to_ell)) ** 0.5
    coeff_xs = (alpha_cum_s_to_t ** 0.5) - coeff_xell * (alpha_cum_s_to_ell ** 0.5)

    return coeff_xell.to(dtype=f8), coeff_xs.to(dtype=f8), std.to(dtype=f8)


def sample_bridge_kernel(
        x_ell: torch.Tensor,
        x_s: torch.Tensor,
        epsilon_net,
        ell: int,
        t: int,
        s: int,
        eta: float = 1.0,
        noise: Optional[torch.Tensor] = None,   # <-- new optional argument
):
    mean, std = bridge_kernel_statistics(x_ell, x_s, epsilon_net, ell, t, s, eta)
    if noise is None:
        z = torch.randn_like(mean)
    else:
        # expect noise to have same shape as mean
        assert noise.shape == mean.shape, f"noise shape {noise.shape} != mean shape {mean.shape}"
        z = noise
    return mean + std * z
