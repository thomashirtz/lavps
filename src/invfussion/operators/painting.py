import torch
from einops import rearrange


class MissingBoxCustom(object): #
    def __init__(self, imshape, device=torch.device("cuda"), generator=None,
                 min_frac=0.1, max_frac=0.6, **kwargs):
        """
        min_frac/max_frac: fraction of image size for each side (width & height).
        Example: min_frac=0.1 => box width >= 0.1*w and height >= 0.1*h
        """
        self.imshape = imshape
        self.device = device
        self.min_frac = float(min_frac)
        self.max_frac = float(max_frac)
        self.update(imshape=imshape, generator=generator)

    def _sample_boxes(self, b, h, w, generator=None):
        device = self.device

        min_bw = max(1, int(self.min_frac * w))
        max_bw = max(min_bw, int(self.max_frac * w))
        min_bh = max(1, int(self.min_frac * h))
        max_bh = max(min_bh, int(self.max_frac * h))

        bw = torch.randint(min_bw, max_bw + 1, (b,), device=device, generator=generator)
        bh = torch.randint(min_bh, max_bh + 1, (b,), device=device, generator=generator)

        # Per-sample maximum valid top-left corner
        max_x1 = (w - bw + 1).clamp_min(1)  # shape (b,)
        max_y1 = (h - bh + 1).clamp_min(1)  # shape (b,)

        # Sample x1, y1 uniformly in [0, max_x1) and [0, max_y1) per-sample
        x1 = (torch.rand(b, device=device, generator=generator) * max_x1).floor().long()
        y1 = (torch.rand(b, device=device, generator=generator) * max_y1).floor().long()

        x2 = x1 + bw
        y2 = y1 + bh
        return x1, x2, y1, y2

    def update(self, imshape=None, generator=None):
        imshape = imshape if imshape is not None else self.imshape
        self.imshape = imshape  # keep in sync if you pass a new shape
        b, c, h, w = imshape

        x1, x2, y1, y2 = self._sample_boxes(b, h, w, generator=generator)

        x_range = torch.arange(w, device=self.device)[None, :]  # (1, w)
        y_range = torch.arange(h, device=self.device)[None, :]  # (1, h)

        mask_zero_x = (x_range >= x1[:, None]) & (x_range < x2[:, None])  # (b, w)
        mask_zero_y = (y_range >= y1[:, None]) & (y_range < y2[:, None])  # (b, h)

        hole = mask_zero_y[:, :, None] & mask_zero_x[:, None, :]  # (b, h, w)
        self.deg_mask = (~hole)[:, None, None, :, :].float()     # (b, 1, 1, h, w)
        self.ndim = 5

    def H(self, x):
        if x.ndim == 4:
            self.ndim = 4
            x = rearrange(x, "b (k c) h w -> b k c h w", c=self.imshape[1])
        else:
            self.ndim = 5
        return rearrange(self.deg_mask * x, "b k c h w -> b k (c h w)")

    def H_pinv(self, y, noise_level=0):
        _, c, h, w = self.imshape
        y = y / (1 + noise_level ** 2)
        _x = self.deg_mask * rearrange(y, "b k (c h w) -> b k c h w", c=c, h=h, w=w)
        if self.ndim == 4:
            return rearrange(_x, "b k c h w -> b (k c) h w")
        return _x


class MissingBox(object):
    def __init__(self, imshape, device=torch.device("cuda"), generator=None, gen_cpu=None, **kwargs):
        self.imshape = imshape
        b, c, h, w = imshape
        box_x = torch.ones((b, w + 1), dtype=torch.float, device=device).multinomial(2,
                                                                                     generator=generator).sort().values
        box_y = torch.ones((b, h + 1), dtype=torch.float, device=device).multinomial(2,
                                                                                     generator=generator).sort().values
        x_range = torch.arange(w, device=device).unsqueeze(0)
        y_range = torch.arange(h, device=device).unsqueeze(0)
        mask_zero_x = (x_range >= box_x[:, 0:1]) & (x_range < box_x[:, 1:2])
        mask_zero_y = (y_range >= box_y[:, 0:1]) & (y_range < box_y[:, 1:2])
        self.deg_mask = torch.logical_not(mask_zero_y[:, :, None].float() * mask_zero_x[:, None, :])[:, None, None, :,
                        :].float()
        self.ndim = 5

    def update(self, imshape=None, generator=None):
        imshape = imshape if imshape is not None else self.imshape
        b, c, h, w = imshape
        device = self.deg_mask.device
        box_x = torch.ones((b, w + 1), dtype=torch.float, device=device).multinomial(2,
                                                                                     generator=generator).sort().values
        box_y = torch.ones((b, h + 1), dtype=torch.float, device=device).multinomial(2,
                                                                                     generator=generator).sort().values
        x_range = torch.arange(w, device=device).unsqueeze(0)
        y_range = torch.arange(h, device=device).unsqueeze(0)
        mask_zero_x = (x_range >= box_x[:, 0:1]) & (x_range < box_x[:, 1:2])
        mask_zero_y = (y_range >= box_y[:, 0:1]) & (y_range < box_y[:, 1:2])
        self.deg_mask = torch.logical_not(mask_zero_y[:, :, None].float() * mask_zero_x[:, None, :])[:, None, None, :,
                        :].float()
        self.ndim = 5

    def H(self, x):
        if x.ndim == 4:
            self.ndim = 4
            x = rearrange(x, "b (k c) h w -> b k c h w", c=self.imshape[1])
        else:
            self.ndim = 5
        return rearrange(self.deg_mask * x, "b k c h w -> b k (c h w)")

    def H_pinv(self, y, noise_level=0):
        _, c, h, w = self.imshape
        y = y / (1 + noise_level ** 2)
        _x = self.deg_mask * rearrange(y, "b k (c h w) -> b k c h w", c=c, h=h, w=w)
        if self.ndim == 4:
            return rearrange(_x, "b k c h w -> b (k c) h w")
        return _x


class CenterBox(MissingBox): #
    """ Added by Léon """

    def __init__(self, imshape, factor, device=torch.device("cuda"), **kwargs):
        """
        imshape: (b, c, h, w)
        factor:  side length s = min(h, w) // factor
        """
        self.imshape = imshape
        self.factor = int(factor)
        if self.factor < 1:
            raise ValueError("factor must be >= 1")
        b, c, h, w = imshape
        self._build_mask(b, c, h, w, device)
        self.ndim = 5  # match parent behavior

    def _build_mask(self, b, c, h, w, device):
        s = max(1, min(h, w) // self.factor)
        # center coordinates
        cx, cy = w // 2, h // 2
        # start (inclusive) and end (exclusive)
        x0 = max(0, min(cx - s // 2, w - 1))
        y0 = max(0, min(cy - s // 2, h - 1))
        x1 = min(x0 + s, w)
        y1 = min(y0 + s, h)

        x_range = torch.arange(w, device=device).unsqueeze(0)  # (1, w)
        y_range = torch.arange(h, device=device).unsqueeze(0)  # (1, h)

        mask_zero_x = (x_range >= x0) & (x_range < x1)  # (1, w)
        mask_zero_y = (y_range >= y0) & (y_range < y1)  # (1, h)

        # 1 outside the box, 0 inside; broadcast over batch/k/channel
        self.deg_mask = torch.logical_not(
            mask_zero_y[:, :, None].float() * mask_zero_x[:, None, :].float()
        ).expand(b, h, w).reshape(b, 1, 1, h, w).float()

    def update(self, imshape=None, **kwargs):
        """
        Rebuild the deterministic center mask for a (possibly) new shape.
        """
        imshape = imshape if imshape is not None else self.imshape
        self.imshape = imshape
        b, c, h, w = imshape
        device = self.deg_mask.device if hasattr(self, "deg_mask") else torch.device("cuda")
        self._build_mask(b, c, h, w, device)
        self.ndim = 5  # keep consistent with parent


class MissingPatches(object):
    def __init__(self, imshape, patch_size=(0, None), p=(0, 0.1), device=torch.device("cuda"), generator=None,
                 gen_cpu=None, **kwargs):
        self.imshape = imshape
        b, c, h, w = imshape
        patch_max = patch_size[1] if patch_size[1] is not None else torch.log2(torch.tensor(min(h, w))).long().item()
        self.patch_size = 2 ** torch.randint(low=patch_size[0], high=1 + patch_max, size=(1,), generator=gen_cpu).item()
        prob = torch.rand((1,), device=device, generator=generator) * (p[1] - p[0]) + p[0]
        while not torch.all(torch.any(mask := (torch.rand((b, 1, 1, h // self.patch_size, w // self.patch_size),
                                                          device=device, generator=generator) < prob
        ).repeat_interleave(self.patch_size, dim=-2).repeat_interleave(self.patch_size, dim=-1), dim=(1, 2, 3, 4))):
            self.patch_size = 2 ** torch.randint(low=patch_size[0], high=1 + patch_max, size=(1,),
                                                 generator=gen_cpu).item()
            prob = torch.randn((1,), device=device, generator=generator) * (p[1] - p[0]) + p[0]
        self.deg_mask = mask.float()
        self.ndim = 5

    def H(self, x):
        if x.ndim == 4:
            self.ndim = 4
            x = rearrange(x, "b (k c) h w -> b k c h w", c=self.imshape[1])
        else:
            self.ndim = 5
        return rearrange(self.deg_mask * x, "b k c h w -> b k (c h w)")

    def H_pinv(self, y, noise_level=0):
        _, c, h, w = self.imshape
        y = y / (1 + noise_level ** 2)
        _x = self.deg_mask * rearrange(y, "b k (c h w) -> b k c h w", c=c, h=h, w=w)
        if self.ndim == 4:
            return rearrange(_x, "b k c h w -> b (k c) h w")
        return _x


class MissingSmallPatches(MissingPatches): #
    def __init__(self, imshape, device=torch.device("cuda"), generator=None,
                 gen_cpu=None, min_p=0.08, max_p=0.10, **kwargs):
        super().__init__(imshape, (0, 0), (min_p, max_p), device, generator, gen_cpu, **kwargs)
