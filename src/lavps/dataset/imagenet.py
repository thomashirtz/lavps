import os
from pathlib import Path

import torch
import torchvision
from torchvision import transforms


class ImageNet(torchvision.datasets.ImageNet):
    def __init__(self, split="train", image_size=256, path="/path/to/imagenet",
                 num_images=None, offset=0, transform=None, one_image_per_class=True, **kwargs):

        default_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor (C, H, W) with values between 0 and 1.
            lambda x_: (x_ - 0.5) * 2,  # Make the value range between -1 and 1.
            transforms.Resize(image_size),  # Resize the shorter side to "size" pixels (aspect ratio preserved)
            transforms.CenterCrop(image_size),  # Crop out the center "size" x "size" region
        ])
        self.transform = transform or default_transform

        super_split = "val" if split == "test" else split
        super().__init__(root=path, split=super_split, transform=default_transform)
        self.num_channels = 3

        if one_image_per_class:
            if split == "val":  # take the first image of each class for val and the last one for test
                self.imgs = [x for i, x in enumerate(self.imgs) if i % 50 == 0]
                self.samples = [x for i, x in enumerate(self.samples) if i % 50 == 0]
                self.targets = [x for i, x in enumerate(self.targets) if i % 50 == 0]
            elif split == "test":
                self.imgs = [x for i, x in enumerate(self.imgs) if i % 50 == 49]
                self.samples = [x for i, x in enumerate(self.samples) if i % 50 == 49]
                self.targets = [x for i, x in enumerate(self.targets) if i % 50 == 49]

        if split in ('test', 'val') and num_images is not None:
            if one_image_per_class:
                assert offset + num_images < 1000
            self.imgs = self.imgs[offset:offset+num_images]
            self.samples = self.samples[offset:offset+num_images]
            self.targets = self.targets[offset:offset+num_images]

        print()

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        image_path = Path(self.imgs[index][0])
        return {
            "image": image,
            "path": str(image_path),
            "name": str(image_path.stem),
            "label": torch.nn.functional.one_hot(torch.tensor([label], device='cpu'), num_classes=1000),
            "label_digit": torch.tensor([label], device='cpu'),
        }


class ImageNet_Xpct(ImageNet):
    def __init__(self, split="train", resolution=256, path="/path/to/imagenet",
                 num_images=None, pct=0.1, **kwargs):
        super().__init__(split=split, resolution=resolution, path=path, num_images=num_images, **kwargs)
        if split == "train" and pct is not None:
            # pct in (0, 1], e.g. 0.1 means ~10%
            if not (0 < pct <= 1):
                raise ValueError(f"pct must be in (0, 1], got {pct}")
            stride = max(1, int(round(1.0 / pct)))  # pct=0.1 → stride=10, pct=0.25 → stride=4
            self.imgs = self.imgs[::stride]
            self.samples = self.samples[::stride]
            self.targets = self.targets[::stride]


class ImageNet_1pct(ImageNet):
    def __init__(self, split="train", resolution=256, path="/path/to/imagenet",
                 num_images=None, **kwargs):
        super().__init__(split=split, resolution=resolution, path=path, num_images=num_images, **kwargs)
        if split == "train":
            self.imgs = [x for i, x in enumerate(self.imgs) if i % 100 < 1]
            self.samples = [x for i, x in enumerate(self.samples) if i % 100 < 1]
            self.targets = [x for i, x in enumerate(self.targets) if i % 100 < 1]


class ImageNet_10pct(ImageNet):
    def __init__(self, split="train", resolution=256, path="/path/to/imagenet",
                 num_images=None, **kwargs):
        super().__init__(split=split, resolution=resolution, path=path, num_images=num_images, **kwargs)
        if split == "train":
            self.imgs = [x for i, x in enumerate(self.imgs) if i % 100 < 10]
            self.samples = [x for i, x in enumerate(self.samples) if i % 100 < 10]
            self.targets = [x for i, x in enumerate(self.targets) if i % 100 < 10]


if __name__ == "__main__":
    os.chdir("..")
    dataset = ImageNet(split="test", path='/path/to/imagenet', num_images=300)
    print(f"Number of images in the dataset: {len(dataset)}")
    dataset_train = ImageNet(split="val", path='/path/to/imagenet', num_images=32)
    print(f"Number of images in the dataset: {len(dataset_train)}")
    ref_image, ref_label = dataset[0]
    print(f"Image shape: {ref_image.shape}, Label shape: {ref_label.shape}")
