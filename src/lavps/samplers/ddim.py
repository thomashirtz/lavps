from typing import Optional

import torch
import tqdm

from lavps.networks import EpsilonNet
from lavps.samplers.utils import bridge_kernel_statistics, sample_bridge_kernel


def ddim_statistics(
        x: torch.Tensor,
        epsilon_net: EpsilonNet,
        t: float,
        t_prev: float,
        eta: float,
        e_t: torch.Tensor = None,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t) # todo here
    return bridge_kernel_statistics(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta
    )


def ddim_step(
        x: torch.Tensor,
        epsilon_net: EpsilonNet,
        t: float,
        t_prev: float,
        eta: float,
        e_t: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,   # <-- pass noise down
        **kwargs,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t, **kwargs)
    return sample_bridge_kernel(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta,
        noise=noise
    )


def ddim(
        initial_noise_sample: torch.Tensor, epsilon_net: EpsilonNet, eta: float = 1.0, **kwargs,
) -> torch.Tensor:
    """
    This function implements the (subsampled) generation from https://arxiv.org/pdf/2010.02502.pdf (eqs 9,10, 12)
    :param initial_noise_sample: Initial "noise"
    :param timesteps: List containing the timesteps. Should start by 999 and end by 0
    :param score_model: The score model
    :param eta: the parameter eta from https://arxiv.org/pdf/2010.02502.pdf (eq 16)
    :return:
    """
    sample = initial_noise_sample
    for i in tqdm.tqdm(range(len(epsilon_net.timesteps) - 1, 1, -1)):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample = ddim_step(
            x=sample,
            epsilon_net=epsilon_net,
            t=t,
            t_prev=t_prev,
            eta=eta,
            **kwargs,
        )
    sample = epsilon_net.predict_x0(sample, epsilon_net.timesteps[1], **kwargs) # todo issue here, missing kwargs at the end

    return epsilon_net.decode(sample) if hasattr(epsilon_net.net, "decode") else sample
