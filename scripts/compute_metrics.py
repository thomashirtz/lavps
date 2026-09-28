import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from natsort import natsorted

from cmmd_pytorch import distance, embedding, io_util
from cmmd_pytorch.main import compute_cmmd as _compute_cmmd
from lavps.paths import MODELS_DIRECTORY
from lavps.utils.metrics import FID
from lavps.utils.metrics import LPIPS, PSNR, SSIM

REFERENCE_PATTERN = "*-reference*"
RECONSTRUCTION_PATTERN = "*-reconstruction*"
OBSERVATION_PATTERN = "*-observation*"

lpips, ssim, psnr = LPIPS(), SSIM(), PSNR()

_FID_CACHE = {}  # module-level

NUM_IMAGES_CMMD = 300
NUM_IMAGES_FID_10K = 10000
NUM_IMAGES_FID_300 = 300


def parse_name(name: str) -> dict: # todo use the parser inside the utils or move this parser inside utils
    """
    Parse a folder-name string of the form:
      key1=value1-key2=value2_with_spaces-key3=True-...
    into a dict:
      { 'key1': typed_value1, 'key2': 'value2 with spaces', 'key3': True, ... }
    """
    result = {}
    # Split on hyphens to get each key=value pair
    for segment in name.split('-'):
        if '=' not in segment:
            continue
        key, raw_val = segment.split('=', 1)
        # Replace underscores with spaces
        val = raw_val.replace('_', ' ')
        # Attempt type conversions: bool, int, float; else leave as string
        if val.lower() in {'true', 'false'}:
            typed = val.lower() == 'true'
        else:
            # Try integer
            int_match = re.fullmatch(r'-?\d+', val)
            float_match = re.fullmatch(r'-?\d+\.\d*', val)
            if int_match:
                typed = int(val)
            elif float_match:
                typed = float(val)
            else:
                typed = val
        result[key] = typed
    return result


def compute_data_stats(
        reconstruction_paths: List[Path] | None = None,
        observation_paths: List[Path] | None = None,
        reference_paths: List[Path] | None = None,
) -> dict[str, float]:
    return {
        'num_reconstructions': len(reconstruction_paths),
        'num_observations': len(observation_paths),
        'num_references': len(reference_paths)
    }


def load_image(path: str) -> torch.Tensor:
    # your existing loader; must return a C×H×W float tensor in [0,1]
    img = Image.open(path).convert('RGB')
    img_torch = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return img_torch * 2 - 1


def compute_basic_metric(
        reconstruction_paths: List[Path],
        observation_paths: List[Path],
        reference_paths: List[Path],
        batch_size: int = 8,
) -> Dict[str, float]:
    """
    Batch through all (recon, ref) pairs, compute LPIPS/PSNR/SSIM/L2, then return their means.
    """
    all_lpips, all_psnr, all_ssim, all_l2 = [], [], [], []

    # iterate in batches
    for i in range(0, len(reconstruction_paths), batch_size):
        batch_recon = reconstruction_paths[i: i + batch_size]
        batch_ref = reference_paths[i: i + batch_size]

        # load & stack
        rec_tensors = torch.stack([load_image(str(p)) for p in batch_recon], dim=0)
        ref_tensors = torch.stack([load_image(str(p)) for p in batch_ref], dim=0)

        # compute metrics
        all_lpips.append(lpips.score(samples=rec_tensors, references=ref_tensors).cpu().numpy())
        all_psnr.append(psnr.score(samples=rec_tensors, references=ref_tensors).cpu().numpy())
        all_ssim.append(ssim.score(samples=rec_tensors, references=ref_tensors).cpu().numpy())

    # concatenate and take means
    all_lpips = np.concatenate(all_lpips)
    all_psnr = np.concatenate(all_psnr)
    all_ssim = np.concatenate(all_ssim)

    return {
        "lpips_mean": float(all_lpips.mean()),
        "psnr_mean": float(all_psnr.mean()),
        "ssim_mean": float(all_ssim.mean()),
        "lpips_std": float(all_lpips.std()),
        "psnr_std": float(all_psnr.std()),
        "ssim_std": float(all_ssim.std()),
    }


@torch.inference_mode()
def _compute_cmmd_fast(ref_dir: Path, eval_dir: Path, *, batch_size=64, max_count=300,
                       cache_dir: Path | str = None) -> float:
    cache_dir = Path(cache_dir) if cache_dir else Path(".")
    # one model reused across calls
    model = embedding.ClipEmbeddingModel(cache_dir)
    # (Optional) if the model exposes a torch module, ensure CUDA:
    # try: model.model.to("cuda"); except Exception: pass

    ref_embs = _embed_dir_cached(ref_dir, model, batch_size, max_count, cache_dir)
    eval_embs = _embed_dir_cached(eval_dir, model, batch_size, max_count, cache_dir)
    return float(distance.mmd(ref_embs, eval_embs).numpy())


def compute_cmmd(
        reconstruction_paths: List[Path] | None = None,
        observation_paths: List[Path] | None = None,
        reference_paths: List[Path] | None = None,
) -> dict[str, float]:
    metric = _compute_cmmd(
        ref_dir=reference_paths[:NUM_IMAGES_CMMD],
        eval_dir=reconstruction_paths[:NUM_IMAGES_CMMD],
        cache_dir=MODELS_DIRECTORY,
    ).item()
    return {"cmmd": metric}


@torch.inference_mode()
def _fingerprint_dir(dirpath: Path, max_count: int) -> str:
    # hash first max_count files’ names + size + mtime
    files = sorted([p for p in Path(dirpath).iterdir() if p.is_file()])[:max_count if max_count > 0 else None]
    h = hashlib.sha256()
    for p in files:
        st = p.stat()
        h.update(str(p).encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()


@torch.inference_mode()
def _embed_dir_cached(dirpath: Path, model, batch_size: int, max_count: int, cache_dir: Path) -> np.ndarray:
    cache_dir = Path(cache_dir) / "cmmd_embeds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _fingerprint_dir(dirpath, max_count)
    cache_file = cache_dir / f"{key}.npy"
    if cache_file.exists():
        return np.load(cache_file).astype("float32")
    embs = io_util.compute_embeddings_for_dir(str(dirpath), model, batch_size, max_count).astype("float32")
    np.save(cache_file, embs)
    return embs


def compute_fid_n(
        reconstruction_paths: List[Path] | None = None,
        observation_paths: List[Path] | None = None,
        reference_paths: List[Path] | None = None,
        *,
        n: int = 10000,
        device: str = "cuda",
) -> dict[str, float]:
    """
    Compute FID using the first n images (or all if n==0).
    Produces a key like {"fid_50000": ...}.
    """
    assert reconstruction_paths is not None and reference_paths is not None

    if n == 0:
        n = min(len(reference_paths), len(reconstruction_paths))
    else:
        n = min(n, len(reference_paths), len(reconstruction_paths))

    fid = get_fid_for_ref(reference_paths, n, device)
    try:
        fid_score = fid.compute_FID(path_imgs=reconstruction_paths[:n])
    except ValueError:
        # Some implementations can throw on numerical issues (imaginary component)
        fid_score = -1.0

    return {f"fid_{n}": float(fid_score)}



def compute_time_per_image_stats(filepath: str, metric_name: str = 'time_per_image') -> Dict[str, float]:
    """
    Read a CSV at `filepath`, compute mean and (sample) std of the `time_per_image` column.
    Returns {'mean': 0.0, 'std': 0.0} on any error or if no numeric data is found.
    """
    default_return_value = {f"{metric_name}_mean": 0.0, f"{metric_name}_std": 0.0}
    try:
        df = pd.read_csv(filepath)
        if metric_name not in df.columns:
            return default_return_value

        s = pd.to_numeric(df["time_per_image"], errors="coerce").dropna()
        if s.empty:
            return default_return_value

        mean_val = float(s.mean())
        std_val = float(s.std(ddof=1)) if len(s) >= 2 else 0.0

        if math.isnan(mean_val) or math.isnan(std_val):
            return default_return_value

        return {f"{metric_name}_mean": mean_val, f"{metric_name}_std": std_val}
    except Exception:
        return default_return_value


def get_fid_for_ref(ref_paths, n, device):
    # Cache key depends on the actual files and n (and mtimes for safety)
    selected = ref_paths[:n] if n > 0 else ref_paths
    h = hashlib.sha256()
    h.update(str(n).encode())
    h.update(str(device).encode())
    for p in selected:
        p = Path(p)
        st = p.stat()
        h.update(str(p).encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    key = h.hexdigest()

    if key not in _FID_CACHE:
        _FID_CACHE[key] = FID(path_real_imgs=selected, device=device)
    return _FID_CACHE[key]


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_file", type=Path, required=False)
    parser.add_argument("-p", "--parent_folder", type=Path, required=True)
    parser.add_argument("--fid_n", type=int, default=300,
                        help="Number of images to use for FID (e.g. 50000). Use 0 for all available.")
    parser.add_argument("--fid_device", type=str, default="cuda",
                        help="Device for FID computation (e.g. cuda, cuda:0, cpu).")
    return parser.parse_args()



if __name__ == '__main__':

    args = get_arguments()
    parent_folder = args.parent_folder

    if args.output_file is not None:
        output_path = args.output_file
    else:
        output_path = parent_folder.with_name(parent_folder.name + ".jsonl")

    fid_n = args.fid_n
    fid_device = args.fid_device

    print(output_path)
    print(parent_folder)

    raw_folder_list = [p for p in parent_folder.iterdir() if p.is_dir()]
    for raw_folder_path in raw_folder_list:
        folder_path = Path(raw_folder_path)

        reconstruction_paths: List[Path] = natsorted(list(folder_path.glob(RECONSTRUCTION_PATTERN)))
        observation_paths: List[Path] = natsorted(list(folder_path.glob(OBSERVATION_PATTERN)))
        reference_paths: List[Path] = natsorted(list(folder_path.glob(REFERENCE_PATTERN)))
        assert len(reconstruction_paths) == len(reference_paths) > 0

        metric_function_list = [
            compute_cmmd,
            compute_data_stats,
            compute_basic_metric,
            # partial(compute_fid_n, n=fid_n, device=fid_device),
        ]

        data = {
            'id': '*',
            'folder_name': folder_path.name,
            'stem': folder_path.stem,
            'parent': folder_path.parent.stem,
            'parent_2': folder_path.parents[1].stem,
        }

        try:
            csv_path = folder_path.parent / f'results-{folder_path.name}.csv'
            data.update(compute_time_per_image_stats(csv_path))
        except Exception:
            print(f'failed to parse {csv_path}')

        for metric_function in metric_function_list:
            data.update(
                metric_function(
                    reconstruction_paths=reconstruction_paths,
                    observation_paths=observation_paths,
                    reference_paths=reference_paths
                )
            )

        data.update(parse_name(folder_path.name))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'a') as f:
            f.write(json.dumps(data) + '\n')
