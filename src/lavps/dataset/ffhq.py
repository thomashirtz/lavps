from pathlib import Path
from typing import Union, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from lavps.paths import REPOSITORY_PATH


def pil_to_tensor(image):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)

class FFHQ(Dataset):
    """
    A PyTorch Dataset for loading FFHQ-style images from a directory.

    Args:
        path (str | Path): Root directory containing image files.
        image_size (int): Desired square size to resize and crop images to.
        offset (int, optional): Number of images to skip at the start. Defaults to 0.
        num_images (int, optional): Maximum number of images to load. Defaults to None (all).
        transform (Callable, optional): Optional transform to apply to each image tensor.
        image_name_list (List[str], optional): Specific filenames to include (overrides offset/num_images).
    """
    def __init__(
        self,
        path: Union[str, Path],
        image_size: int = 64,
        offset: int = 0,
        num_images: int | None = None,
        transform: Callable | None  = None,
        image_name_list: list[str] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(path)

        # Gather image files
        all_images = sorted(self.root.glob("*.png"))

        # Filter by explicit name list if provided
        if image_name_list is not None:
            selected = [self.root / name for name in image_name_list]
        else:
            selected = all_images[offset : offset + num_images if num_images is not None else None]

        self.image_paths = selected
        if len(self.image_paths) == 0:
            raise ValueError("No images found with the given parameters.")

        # Default transform: to tensor, scale [-1,1], resize & center crop
        default_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor (C, H, W) with values between 0 and 1.
            lambda x_: (x_ - 0.5) * 2,  # Make the value range between -1 and 1.
            transforms.Resize(image_size),  # Resize the shorter side to "size" pixels (aspect ratio preserved)
            transforms.CenterCrop(image_size),  # Crop out the center "size" x "size" region
        ])

        self.transform = transform or default_transform
        self.image_size = image_size
        print(f'{len(self.image_paths)=}')

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)

        return {
            "image": image,
            "path": str(image_path),
            "name": str(image_path.stem),
            "label": torch.zeros((0,), device="cpu"),
        }

if __name__ == '__main__':
    ffhq = FFHQ(
        name="ffhq_val",
    path='/path/to/ffhq',
    num_images=10,
    )
    t = ffhq[0]
    print()
