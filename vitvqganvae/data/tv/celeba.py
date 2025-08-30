from einops import rearrange
from torch import Tensor
from torchvision.datasets import CelebA
from torchvision import transforms
from torchvision.utils import make_grid

from ..utils import ConcatDataset

import torch

def make_grid_celeba(original: Tensor, reconstructed: Tensor, nrow: int | None = None) -> Tensor:
    imgs_and_recons = torch.stack((original, reconstructed), dim=0)
    imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

    imgs_and_recons = imgs_and_recons.detach().cpu().float()
    return make_grid(imgs_and_recons, nrow=nrow) + 0.5

def get_celeba(root: str | None = None, download: bool = True) -> CelebA:

    train_ds = CelebA(
        root=root,
        split="train",
        download=download,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ])
    )

    valid_ds = CelebA(
        root=root,
        split="valid",
        download=download,
        transform=transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ])
    )

    test_ds = CelebA(
        root=root,
        split="test",
        download=download,
        transform=transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ])
    )

    train_ds = ConcatDataset([train_ds, valid_ds])

    return train_ds, test_ds