from einops import rearrange
from torch import Tensor
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torchvision.utils import make_grid

import torch

def make_grid_cifar(original: Tensor, reconstructed: Tensor, nrow: int | None = None) -> Tensor:
    imgs_and_recons = torch.stack((original, reconstructed), dim=0)
    imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

    imgs_and_recons = imgs_and_recons.detach().cpu().float()
    return make_grid(imgs_and_recons, nrow=nrow) + 0.5

def make_grid_cifar10(original: Tensor, reconstructed: Tensor, nrow: int | None = None) -> Tensor:
    return make_grid_cifar(original, reconstructed, nrow=nrow)

def make_grid_cifar100(original: Tensor, reconstructed: Tensor, nrow: int | None = None) -> Tensor:
    return make_grid_cifar(original, reconstructed, nrow=nrow)

def get_cifar10(root: str | None = None, download: bool = True) -> CIFAR10:

    train_ds = CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ])
    )

    valid_ds = CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ])
    )

    return train_ds, valid_ds

def get_cifar100(root: str | None = None, download: bool = True) -> CIFAR100:
    train_ds = CIFAR100(
        root=root,
        train=True,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ])
    )

    valid_ds = CIFAR100(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ])
    )

    return train_ds, valid_ds