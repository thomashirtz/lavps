import torch

import dnnlib
from invfussion.models.hdit import HDiTInferenceModelInvFussion, HDiTInferenceModelPalette
from lavps.samplers.utils import bridge_kernel_statistics


def get_sigma_and_scale(epsilon_net, timesteps):
    acp = epsilon_net.alphas_cumprod[timesteps] / epsilon_net.alphas_cumprod[
        epsilon_net.timesteps[0].int()]
    sigma = (1 - acp) ** 0.5
    scale = acp ** 0.5
    return sigma, scale


def expand_like(bvec, like):
    # turn (B,) into (B,1,1,1) (or generally (B,1,1,...)) to match `like`
    return bvec.view(bvec.shape[0], *([1] * (like.dim() - 1))).to(device=like.device, dtype=like.dtype)


class ResidualInferenceModel(torch.nn.Module):
    def __init__(self, residual_nn, epsilon_net):
        super().__init__()
        self.residual_nn = residual_nn
        self.epsilon_net = epsilon_net

    def predict_residual(self, x0, xt, y, H, timesteps_s, timesteps_t, **kwargs):
        raise NotImplementedError

    def forward(self, x0, xt, y, H, timesteps_s, timesteps_t, return_dict=False, class_labels=None):
        kwargs = {'class_cond': class_labels.to(device=x0.device, dtype=torch.float32)} if class_labels is not None else {}
        delta_vmean, delta_vlog_std = self.predict_residual(x0, xt, y, H, timesteps_s, timesteps_t, **kwargs)

        mean_prior, std_prior = bridge_kernel_statistics(
            xt, x0, self.epsilon_net, timesteps_t, timesteps_s, 0, eta=1.0
        )
        mean_prior = mean_prior.float()
        std_prior = std_prior.float()
        std_prior = std_prior.reshape(std_prior.shape[0], 1, 1, 1)
        log_std_prior = std_prior.log() * torch.ones_like(xt)
        vmean = mean_prior.detach() + delta_vmean
        vlog_std = log_std_prior.detach() + delta_vlog_std
        if not return_dict:
            return vmean, vlog_std
        return {
            "vmean": vmean,
            "vlog_std": vlog_std,
            "mean_prior": mean_prior.detach(),
            "log_std_prior": log_std_prior.detach()
        }


class ResidualInferenceModelHDiTInvFussion(ResidualInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.residual_nn, HDiTInferenceModelInvFussion)

    def predict_residual(self, x0, xt, y, H, timesteps_s, timesteps_t, **kwargs):
        sigma_t, scale_t = get_sigma_and_scale(self.epsilon_net, timesteps_t)
        sigma_s, scale_s = get_sigma_and_scale(self.epsilon_net, timesteps_s)
        scale_t_ = expand_like(scale_t, xt)
        return self.residual_nn(x_t=xt / scale_t_, sigma_t=sigma_t / scale_t, x0=x0, sigma_s=sigma_s / scale_s, degradation=H, y=y, **kwargs)


class ResidualInferenceModelHDiTPalette(ResidualInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.residual_nn, HDiTInferenceModelPalette)

    def predict_residual(self, x0, xt, y, H, timesteps_s, timesteps_t, **kwargs):
        y_adj = H.H_pinv(y).reshape(x0.shape)
        sigma_t, scale_t = get_sigma_and_scale(self.epsilon_net, timesteps_t)
        sigma_s, scale_s = get_sigma_and_scale(self.epsilon_net, timesteps_s)
        scale_t_ = expand_like(scale_t, xt)
        return self.residual_nn(x_t=xt / scale_t_, sigma_t=sigma_t / scale_t, x0=x0, sigma_s=sigma_s / scale_s, y_adj=y_adj, **kwargs)


def get_inference_model(
        epsilon_net,
        inference_model_type: str,
        resolution: int,
        device: torch.device,
        path: str | None = None,
        set_eval_mode: bool = False,
):
    if 'imagenet' in inference_model_type:
        label_dim = 1000
        data_name = f"ImageNet{resolution}"
    elif 'ffhq' in inference_model_type:
        label_dim = 0
        data_name = f"FFHQ{resolution}"
    else:
        raise NotImplementedError

    if 'invfussion' in inference_model_type:
        network_kwargs = dnnlib.EasyDict(
            class_name="invfussion.models.hdit.HDiTInferenceModelInvFussion",
            joint=True,
            in_mult=3,
            out_mult=2,
        )
        residual_class = ResidualInferenceModelHDiTInvFussion
    elif 'palette' in inference_model_type:
        network_kwargs = dnnlib.EasyDict(
            class_name="invfussion.models.hdit.HDiTInferenceModelPalette",
            joint=False,
            in_mult=3,
            out_mult=2,
        )
        residual_class = ResidualInferenceModelHDiTPalette
    else:
        raise NotImplementedError

    interface_kwargs = dnnlib.EasyDict(
        img_resolution=resolution,
        img_channels=3,
        label_dim=label_dim,
        data=data_name,
    )

    residual_nn = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    inference_model = residual_class(residual_nn, epsilon_net)

    if path is not None:
        ckpt = torch.load(path, map_location=device)
        inference_model.load_state_dict(ckpt["model_state_dict"])

    if set_eval_mode:
        inference_model.eval()
        for p in inference_model.parameters():
            p.requires_grad = False

    return inference_model