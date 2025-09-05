from torch.utils.data import Dataset
from einops import rearrange
from torch import Tensor
from torchvision import transforms
from torchvision.utils import make_grid
from ..utils import ConcatDataset

from glob import glob
from PIL import Image

import torch
import os


def make_grid_imagenet(original: Tensor, reconstructed: Tensor, nrow: int | None = None) -> Tensor:
    imgs_and_recons = torch.stack((original, reconstructed), dim=0)
    imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

    imgs_and_recons = imgs_and_recons.detach().cpu().float()
    return make_grid(imgs_and_recons, nrow=nrow) + 0.5

def denorm_imagenet(x: Tensor) -> Tensor:
    return x + 0.5


class ImageNet(Dataset):
    def __init__(self, root: str, split: str = 'train', transform: transforms.Compose | None = None):
        self._root = root
        self._split = split
        self._transform = transform

        if self._split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of 'train', 'val', or 'test'")
    
        self._data_dir = f"{self._root}/{self._split}"
        if not os.path.exists(self._data_dir):
            raise ValueError(f"Directory {self._data_dir} does not exist")

        if self._split == 'train':
            self._image_paths = glob(f"{self._data_dir}/*/*.JPEG")
        else:
            self._image_paths = glob(f"{self._data_dir}/*.JPEG")

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> Tensor:
        img = Image.open(self._image_paths[index]).convert('RGB')
        if self._transform:
            img = self._transform(img)
        return img


def get_imagenet(root: str | None = None, image_size: int = 64) -> ImageNet:

    assert root is not None, "Please provide the path to the ImageNet dataset root directory"

    if image_size not in [64, 128, 256]:
        raise ValueError("image_size should be one of 64, 128, or 256 for ImageNet dataset")

    train_ds = ImageNet(
        root=root,
        split="train",
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ])
    )

    valid_ds = ImageNet(
        root=root,
        split="val",
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ])
    )

    test_ds = ImageNet(
        root=root,
        split="test",
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ])
    )

    train_ds = ConcatDataset([train_ds, valid_ds])

    return train_ds, test_ds