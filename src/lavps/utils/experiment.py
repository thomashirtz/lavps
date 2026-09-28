import gc
import hashlib
import os
import random
from pathlib import Path

import PIL
import numpy as np
import torch
from PIL import Image


def get_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype)
    return dtype


def get_true_device_id(device_id: int = 0):
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')

    # If no environment variable is set, return the provided device_id.
    if visible_devices is None:
        return device_id

    # Try to treat the environment variable as an integer first.
    try:
        true_device = int(visible_devices)
        return true_device
    except ValueError:
        # Not a single integer, assume it's a comma-separated list.
        devices = [d.strip() for d in visible_devices.split(',')]
        try:
            # Get the device corresponding to the provided index.
            selected_device = devices[device_id]
            # Optionally, convert to int if desired.
            try:
                selected_device = int(selected_device)
            except ValueError:
                # Keep as string if conversion fails.
                pass
            return selected_device
        except IndexError:
            raise ValueError(f"Device index {device_id} out of range for CUDA_VISIBLE_DEVICES: {devices}")


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def save_im(x, save_path, title=None):
    x = x.clamp(min=-1, max=1)
    sample = x.squeeze(0).cpu().permute(1, 2, 0)
    sample = (sample + 1.0) * 127.5
    sample = sample.float().numpy().astype(np.uint8)
    img_pil = PIL.Image.fromarray(sample)
    img_pil.save(save_path)
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.imshow(img_pil)
    # ax.set_title(title)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # fig.savefig(save_path)
    # plt.close()


def load_pil_image(image_path: str):
    image_path_base = Path(image_path)

    # Try possible extensions
    for ext in ['.jpg', '.png']:
        image_path = image_path_base.with_suffix(ext)
        if image_path.exists():
            return Image.open(image_path)
    else:
        raise FileNotFoundError(f"Image file not found for {image_path_base} with .jpg or .png extension.")


def seed_from_name(name: str, *, salt: str = "", nbytes: int = 4) -> int:
    """
    Deterministic seed from a string.
    - nbytes=4 -> 32-bit seed (0 .. 2^32-1), good for torch / numpy.
    """
    if not isinstance(name, str):
        name = str(name)

    h = hashlib.blake2b((salt + name).encode("utf-8"), digest_size=nbytes).digest()
    return int.from_bytes(h, byteorder="big", signed=False)


def clean_cache():
    # https://stackoverflow.com/a/74012970
    # https://stackoverflow.com/a/61697029
    gc.collect()
    # Empty the cache
    torch.cuda.empty_cache()
    # Set no grad and empty the cache
    with torch.no_grad():
        torch.cuda.empty_cache()
