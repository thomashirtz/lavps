from typing import List, Any

import torch
from diffusers import DDPMPipeline, DDPMScheduler

import dnnlib
from lavps.paths import MODELS_DIRECTORY


class UnetHG(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t):
        if torch.tensor(t).dim() == 0:
            t = torch.tensor([t])
        return self.unet(x, t).sample


class EpsilonNet(torch.nn.Module):
    def __init__(self, net, alphas_cumprod, timesteps):
        super().__init__()
        self.net = net
        self.register_buffer("timesteps", timesteps.to(torch.long),persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod,persistent=False)
        self.register_buffer("acp_f8", alphas_cumprod.to(torch.float64),persistent=False)

        self.timesteps.requires_grad = False
        self.acp_f8.requires_grad = False
        self.alphas_cumprod.requires_grad = False

    def forward(self, x, t, **kwargs):
        t_tensor = torch.as_tensor(t, device=x.device, dtype=torch.long)
        return self.net(x, t_tensor, **kwargs)

    def predict_x0(self, x, t, **kwargs):
        t = torch.as_tensor(t, device=self.alphas_cumprod.device, dtype=torch.long)
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0].int()]
        acp_t = acp_t.view(-1, 1, 1, 1)
        return (x - (1 - acp_t) ** 0.5 * self.forward(x, expand_like(t, x).long(), **kwargs)) / (acp_t ** 0.5)

    def score(self, x, t, **kwargs):
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0]]
        return -self.forward(x, t, **kwargs) / (1 - acp_t) ** 0.5

    def decode(self, z):
        return self.net.decode(z)

    def differentiable_decode(self, z):
        return self.net.differentiable_decode(z)


def expand_like(bvec, like):
    # turn (B,) into (B,1,1,1) (or generally (B,1,1,...)) to match `like`
    if bvec.ndim == 0:
        return bvec
    return bvec.view(bvec.shape[0], *([1] * (like.dim() - 1))).to(device=like.device, dtype=like.dtype)


def load_epsilon_network(model_id: str, num_steps: int, device: str | torch.device):
    hf_models = {
        "celebahq": "google/ddpm-celebahq-256",
    }
    if model_id in hf_models:
        hf_id = "google/ddpm-celebahq-256"
        pipeline = DDPMPipeline.from_pretrained(hf_id, cache_dir=MODELS_DIRECTORY).to(device)
        model = pipeline.unet
        model = model.requires_grad_(False)
        model = model.eval()

        timesteps = torch.linspace(0, 999, num_steps).long()
        alphas_cumprod = pipeline.scheduler.alphas_cumprod.clip(1e-6, 1)
        alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

        return EpsilonNet(
            net=UnetHG(model), alphas_cumprod=alphas_cumprod, timesteps=timesteps
        )
    raise NotImplementedError(f"Choose correct model_id in load_epsilon_net, currently it is {model_id}")


class SDWrapper(torch.nn.Module):
    def __init__(self, pipeline):
        super().__init__()

        self.pipeline = pipeline
        self.unet, self.vae = pipeline.unet, pipeline.vae
        self.prompt = ""
        self.use_cfg = False

        # these should be freezed as they aren't trainable
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def forward(self, x, t):
        prompt_embeds, uncond_prompt_embeds, *others = self.pipeline.encode_prompt(
            prompt=self.prompt,
            device=x.device,
            num_images_per_prompt=x.shape[0],
            do_classifier_free_guidance=self.use_cfg,
        )
        if self.use_cfg:
            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])
            x = torch.cat([x] * 2)

        added_cond_kwargs = SDWrapper._get_cond_kwargs(others, self.use_cfg)
        pred = self.unet(
            x,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        return pred

    def differentiable_decode(self, z, force_float32: bool = True):
        # force decoding to be in float32 to avoid overflow
        vae = self.vae.to(torch.float32) if force_float32 else self.vae
        z = z.to(torch.float32) if force_float32 else z

        return vae.decode(z / vae.config.scaling_factor).sample

    @torch.no_grad()
    def decode(self, z, force_float32: bool = True):
        return self.differentiable_decode(z, force_float32)

    @staticmethod
    def _get_cond_kwargs(other_prompt_embeds: List[torch.Tensor], use_cfg: bool):
        pooled_prompt_embeds, negative_pooled_prompt_embeds = other_prompt_embeds

        # additional image-based embeddings
        add_time_ids = SDWrapper._get_time_ids(pooled_prompt_embeds)
        negative_add_time_ids = add_time_ids

        add_text_embeds = pooled_prompt_embeds
        if use_cfg:
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds]
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids])

        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }
        return added_cond_kwargs

    @staticmethod
    def _get_time_ids(prompt_embeds: torch.Tensor):
        # NOTE these were deduced from pipeline.__call__ of diffuser v0.27.2
        # and are so far valid for sdxl1.0
        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)

        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor(
            prompt_embeds.shape[0] * [add_time_ids], dtype=prompt_embeds.dtype
        )
        return add_time_ids


class HDiTEpsilonNet(EpsilonNet):
    def __init__(
            self,
            net,
            alphas_cumprod,
            timesteps,
            device,
    ):
        self.device = device
        super().__init__(net=net, timesteps=timesteps, alphas_cumprod=alphas_cumprod)

    def forward(self, x, t, H=None, y=None, **kwargs):
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0].int()]
        x_input = x / acp_t ** 0.5
        sigma = (1 - acp_t) ** 0.5 / acp_t ** 0.5
        if sigma.ndim == 0 or sigma.shape[0] != x.shape[0]:
            sigma = sigma.expand(x.shape[0]) # todo not sure it is the best way to do it, maybe use expland_like that i already wrote
            # we are inconsistent with the input shapes so it makes it very difficult to handle
        d_theta = self.net(x_input, sigma, H=H, y=y, **kwargs)
        f_theta = (x - d_theta * acp_t ** 0.5) / (1 - acp_t) ** 0.5
        return f_theta


def load_network(
        path: str,
        network_kwargs: dict[str, Any],
        interface_kwargs: dict[str, Any],
        device: torch.device,
        fp16: bool = True,
):
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    from invfussion.precond import Precond
    net = Precond(net, use_fp16=fp16, **interface_kwargs).to(device)
    m, u = net.load_state_dict(torch.load(path, map_location=device, weights_only=False)["ema"])
    assert len(m) == 0 and len(u) == 0
    return net


def load_hdit_epsilon_network(
        path: str,
        num_steps: int,
        device: torch.device,
        network_kwargs: dict[str, Any],
        interface_kwargs: dict[str, Any],
        fp16: bool = True,
        **kwargs,
):
    print(f'invfussion.net.load_invfussion_epsilon_network kwargs: {kwargs}')
    net = load_network(path, network_kwargs, interface_kwargs, fp16=fp16, device=device)
    net = net.eval()
    for p in net.parameters():
        p.requires_grad = False
    scheduler = DDPMScheduler()
    timesteps = torch.linspace(0, 999, num_steps).long()
    alphas_cumprod = scheduler.alphas_cumprod.clip(1e-6, 1)
    alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])
    return HDiTEpsilonNet(net, alphas_cumprod=alphas_cumprod, timesteps=timesteps, device=device)
