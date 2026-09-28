from __future__ import annotations

from typing import Optional

import numpy as np
import torch

import cmmd_pytorch.embedding as embedding
import cmmd_pytorch.io_util as io_util


_SIGMA_IMG: float = 10.0   # RBF σ for image embeddings (x)
_SIGMA_OBS: float = 1.0    # RBF σ for observation/conditioning embeddings (y)
_SCALE: float      = 1000.0  # multiplicative so scores are in a readable range


def _gaussian_rbf(a: torch.Tensor, b: torch.Tensor, gamma: float) -> torch.Tensor:
    """Pairwise Gaussian RBF ‖a-b‖² kernel computed memory‑efficiently.

    a: (N, D) tensor; b: (M, D) tensor  →  (N, M) Gram matrix.
    """
    a_sq = (a * a).sum(dim=1, keepdim=True)      # (N,1)
    b_sq = (b * b).sum(dim=1, keepdim=True).T    # (1,M)
    dist = a_sq + b_sq - 2.0 * a @ b.T           # broadcasting
    return torch.exp(-gamma * dist)


# =============================================================================
#   Un‑conditional CMMD (unchanged math; rewritten in Torch for speed)
# =============================================================================

def _cmmd_from_embeddings(real: np.ndarray, gen: np.ndarray, sigma_img: float) -> float:
    """CMMD between two embedding clouds (no batching needed)."""
    x = torch.from_numpy(real)
    y = torch.from_numpy(gen)
    gamma = 1.0 / (2.0 * sigma_img ** 2)

    k_xx = _gaussian_rbf(x, x, gamma).mean()
    k_yy = _gaussian_rbf(y, y, gamma).mean()
    k_xy = _gaussian_rbf(x, y, gamma).mean()

    return float(_SCALE * (k_xx + k_yy - 2.0 * k_xy))

# -----------------------------------------------------------------------------
#   Conditional CMMD via product‑kernel: k_joint((y,x),(y',x')) = kY * kX
# -----------------------------------------------------------------------------

def _ccmmd_from_embeddings(
    obs_emb: np.ndarray,   # φ(y)
    real_emb: np.ndarray,  # ψ(x)
    gen_emb: np.ndarray,   # ψ( x̂ )
    sigma_obs: float,
    sigma_img: float,
) -> float:
    if not (obs_emb.shape[0] == real_emb.shape[0] == gen_emb.shape[0]):
        raise ValueError("obs, real, gen must have the same number of samples (aligned triplets).")

    o = torch.from_numpy(obs_emb)
    r = torch.from_numpy(real_emb)
    g = torch.from_numpy(gen_emb)

    assert obs_emb.shape == real_emb.shape == gen_emb.shape
    print("σ_obs, σ_img:", sigma_obs, sigma_img)
    print("mean(kY):", _gaussian_rbf(o, o, 1 / (2 * sigma_obs ** 2)).mean().item())
    print("mean(kX real-real):", _gaussian_rbf(r, r, 1 / (2 * sigma_img ** 2)).mean().item())
    print("mean(kX real-gen):", _gaussian_rbf(r, g, 1 / (2 * sigma_img ** 2)).mean().item())

    # RBF γ parameters
    gamma_obs = 1.0 / (2.0 * sigma_obs ** 2)
    gamma_img = 1.0 / (2.0 * sigma_img ** 2)

    # Kernel blocks
    kY = _gaussian_rbf(o, o, gamma_obs)        # (N,N)
    kRR = _gaussian_rbf(r, r, gamma_img)
    kGG = _gaussian_rbf(g, g, gamma_img)
    kRG = _gaussian_rbf(r, g, gamma_img)

    # Product kernels in joint space
    k_obs_real = kY * kRR
    k_obs_gen  = kY * kGG
    k_cross    = kY * kRG

    n = kY.size(0)
    inv_n_n1 = 1.0 / (n * (n - 1))
    term_rr = (k_obs_real.sum() - k_obs_real.trace()) * inv_n_n1
    term_gg = (k_obs_gen.sum()  - k_obs_gen.trace())  * inv_n_n1
    term_rg = k_cross.mean()

    mmd2 = term_rr + term_gg - 2.0 * term_rg
    return float(_SCALE * mmd2)

# =============================================================================
#   Public helpers that mirror original API
# =============================================================================

def compute_cmmd(
    ref_dir: str,
    eval_dir: str,
    ref_embed_file: Optional[str] = None,
    *,
    batch_size: int = 32,
    max_count: int = -1,
    sigma_img: float = _SIGMA_IMG,
) -> float:
    """Drop‑in replacement for the original CMMD routine (no flags, no CLI).

    Parameters
    ----------
    ref_dir : directory with ground‑truth images (ignored if `ref_embed_file` given)
    eval_dir: directory with images to evaluate (reconstructions or generations)
    ref_embed_file: optional `.npy` file with *pre‑computed* embeddings of the
        reference set (useful to skip recomputation).
    batch_size / max_count / sigma_img: same meaning as before.
    """
    embedding_model = embedding.ClipEmbeddingModel()

    # ---- get embeddings ------------------------------------------------------
    if ref_embed_file is not None:
        real_emb = np.load(ref_embed_file).astype("float32")
    else:
        real_emb = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype("float32")

    gen_emb = io_util.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")

    return _cmmd_from_embeddings(real_emb, gen_emb, sigma_img)


def compute_ccmmd(
    obs_dir: str,
    ref_dir: str,
    eval_dir: str,
    ref_embed_file: Optional[str] = None,
    *,
    batch_size: int = 32,
    max_count: int = -1,
    sigma_obs: float = _SIGMA_OBS,
    sigma_img: float = _SIGMA_IMG,
) -> float:
    """Conditional CMMD using *product‑kernel* joint MMD.

    `obs_dir` must contain the observation images (y) that *align index‑wise*
    with both `ref_dir` (ground‑truth x) and `eval_dir` (reconstructed x̂).
    """
    embedding_model = embedding.ClipEmbeddingModel()

    # ---- embeddings ----------------------------------------------------------
    obs_emb = io_util.compute_embeddings_for_dir(obs_dir, embedding_model, batch_size, max_count).astype("float32")

    if ref_embed_file is not None:
        real_emb = np.load(ref_embed_file).astype("float32")
    else:
        real_emb = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype("float32")

    gen_emb = io_util.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")

    return _ccmmd_from_embeddings(obs_emb, real_emb, gen_emb, sigma_obs, sigma_img)