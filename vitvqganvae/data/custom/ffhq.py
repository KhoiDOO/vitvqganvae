from torch.utils.data import Dataset
from einops import rearrange
from torch import Tensor
from torchvision import transforms
from torchvision.utils import make_grid
from ..utils import random_split

from glob import glob
from PIL import Image

import torch
import os


def make_grid_ffhq(original: Tensor, reconstructed: Tensor, nrow: int | None = None) -> Tensor:
    imgs_and_recons = torch.stack((original, reconstructed), dim=0)
    imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

    imgs_and_recons = imgs_and_recons.detach().cpu().float()
    return make_grid(imgs_and_recons, nrow=nrow) + 0.5

def denorm_ffhq(x: Tensor) -> Tensor:
    return x + 0.5


class FFHQ(Dataset):
    def __init__(self, root: str, transform: transforms.Compose | None = None):
        super().__init__()

        self._root = root
        self._transform = transform

        if not os.path.exists(self._root):
            raise ValueError(f"Directory {self._root} does not exist")

        self._image_paths = glob(f"{self._root}/*.png")
        if not self._image_paths:
            raise ValueError(f"No images found in {self._root}")

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> Tensor:
        img = Image.open(self._image_paths[index]).convert('RGB')
        if self._transform:
            img = self._transform(img)
        return img

def get_ffhq(root: str | None = None, image_size: int = 64, split: list[int] = [60000, 10000]) -> random_split:

    assert root is not None, "Please provide the path to the FFHQ dataset root directory"

    if image_size not in [64, 128, 256, 512, 1024]:
        raise ValueError("image_size should be one of 64, 128, 256, 512, or 1024 for FFHQ dataset")

    if len(split) != 2:
        raise ValueError("split should be a list of two integers representing the lengths of the train and val sets")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = FFHQ(root=root, transform=transform)

    if sum(split) != len(dataset):
        raise ValueError(f"Sum of split lengths {sum(split)} does not equal the length of the dataset {len(dataset)}")

    train_ds, val_ds = random_split(dataset, split)

    return train_ds, val_ds