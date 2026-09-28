import torch

from invfussion.operators.base import H_functions


class SuperResolution(H_functions): #
    """
    Down-samples by `ratio` with PyTorch interpolate; up-sampling is the
    (approximate) pseudo-inverse.

    Now supports inputs with shape B1,...,BN,C,H,W (any number of leading batch dims).
    """

    def __init__(self, imshape, ratio=4, mix_color=False, device=torch.device("cuda"), generator=None, gen_cpu=None,
                 **kwargs):
        super().__init__()
        # --- CHANGED: read (C,H,W) from the END so extra batch dims are ok
        channels, img_dim, w = imshape[-3:]
        assert img_dim % ratio == 0
        # (keep the square assumption exactly as before)
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio ** 2] * ratio ** 2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.V_small = self.V_small  # unchanged
        self.Vt_small = self.V_small.transpose(0, 1)

    # --- NEW: tiny helpers to flatten/restore leading batch dims (fewest changes elsewhere)
    def _flatten_batch_any(self, x):
        """Returns x2d (B*, -1) and the leading batch shape for restoration."""
        if x.dim() >= 3 and x.shape[-3:] == (self.channels, self.img_dim, self.img_dim):
            # spatial input (..., C, H, W) → 2D
            batch_shape = x.shape[:-3]
            x2d = x.reshape(-1, self.channels * self.img_dim ** 2)
        else:
            # already vector form (..., C*H*W) or (..., C*y*y)
            batch_shape = x.shape[:-1]
            x2d = x.reshape(-1, x.shape[-1])
        return x2d, batch_shape

    def _restore_batch(self, x2d, batch_shape):
        return x2d.reshape(*batch_shape, x2d.shape[-1])

    def V(self, vec):
        # --- CHANGED: flatten all batch dims first
        vec, batch_shape = self._flatten_batch_any(vec)
        # reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim ** 2, self.ratio ** 2, device=vec.device,
                              dtype=vec.dtype)
        patches[:, :, :, 0] = temp[:, :self.channels * self.y_dim ** 2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio ** 2 - 1):
            patches[:, :, :, idx + 1] = temp[:, (self.channels * self.y_dim ** 2 + idx)::self.ratio ** 2 - 1].view(
                vec.shape[0], self.channels, -1)
        # multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio ** 2, 1)).reshape(vec.shape[0],
                                                                                              self.channels, -1,
                                                                                              self.ratio ** 2)
        # repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        # --- CHANGED: restore leading batch dims
        return self._restore_batch(recon, batch_shape)

    def Vt(self, vec):
        # --- CHANGED: flatten all batch dims first
        vec, batch_shape = self._flatten_batch_any(vec)
        # extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        # multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio ** 2, 1)).reshape(vec.shape[0],
                                                                                               self.channels, -1,
                                                                                               self.ratio ** 2)
        # reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim ** 2, device=vec.device, dtype=vec.dtype)
        recon[:, :self.channels * self.y_dim ** 2] = patches[:, :, :, 0].view(vec.shape[0],
                                                                              self.channels * self.y_dim ** 2)
        for idx in range(self.ratio ** 2 - 1):
            recon[:, (self.channels * self.y_dim ** 2 + idx)::self.ratio ** 2 - 1] = patches[:, :, :, idx + 1].view(
                vec.shape[0], self.channels * self.y_dim ** 2)
        # --- CHANGED: restore leading batch dims
        return self._restore_batch(recon, batch_shape)

    def U(self, vec):
        # --- CHANGED: keep original shape; scale in 2D then restore
        orig_shape = vec.shape
        x2d = vec.reshape(-1, orig_shape[-1])
        out = self.U_small[0, 0] * x2d
        return out.reshape(orig_shape)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        # --- CHANGED: same as U
        orig_shape = vec.shape
        x2d = vec.reshape(-1, orig_shape[-1])
        out = self.U_small[0, 0] * x2d
        return out.reshape(orig_shape)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim ** 2)

    def add_zeros(self, vec):
        # --- CHANGED: accept any leading batch dims
        reshaped = vec.clone().reshape(-1, vec.shape[-1])
        temp = torch.zeros((reshaped.shape[0], reshaped.shape[1] * self.ratio ** 2), device=vec.device, dtype=vec.dtype)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        # --- CHANGED: slice last dim, not assuming 2D
        return self.U(singulars * temp[..., :singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        # --- CHANGED: slice last dim, not assuming 2D
        return self.V(self.add_zeros(singulars * temp[..., :singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        vec_reshaped, batch_shape = self._flatten_batch_any(vec)
        temp = self.Ut(vec_reshaped)
        singulars = self.singulars()

        # NOTE: Handle the case when H is rank deficit
        inv_singulars = 1 / singulars
        inv_singulars[singulars == 0.] = 0.

        # --- CHANGED: slice last dim, not assuming 2D
        temp_flat = temp.reshape(-1, temp.shape[-1])
        temp_flat[:, :singulars.shape[0]] = temp_flat[:, :singulars.shape[0]] * inv_singulars
        temp = temp_flat.reshape(*temp.shape)
        output = self.V(self.add_zeros(temp))
        return output.reshape(*batch_shape, self.channels, self.img_dim, self.img_dim)
