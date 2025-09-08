from torch.utils.data import Dataset
from torch import Tensor

from datasets import load_dataset

from ..utils import random_split

from scipy.spatial.transform import Rotation

import numpy as np
import torch
import random


class Mesh500(Dataset):
    def __init__(self, root: str, num_points: int = 1024, augment: bool = True):
        super().__init__()

        if num_points not in [1024, 4096]:
            raise ValueError("num_points should be one of 1024 or 4096 for Mesh500 dataset")

        self._root = root
        self._augment = augment
        self._dataset = load_dataset(f"kohido/mesh500_{num_points}pts", cache_dir=self._root)['train']['points']

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tensor:
        points = self._dataset[index]
        points: np.ndarray = np.array(points) # (1024, 3)
        if self._augment:
            axis = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
            radian = np.pi / 180 * random.uniform(0, 360)
            rotation = Rotation.from_rotvec(radian * np.array(axis))
            points = rotation.apply(points)
        points: Tensor = torch.from_numpy(points) 
        points = points.float()
        points = points.permute(1, 0)
        return points


def get_mesh500(root: str | None = None, num_points: int = 1024, split: float = 0.8, augment: bool = True) -> tuple[Mesh500, Mesh500]:
    assert root is not None, "Please provide the path to the Mesh500 dataset root directory"

    dataset = Mesh500(root=root, num_points=num_points, augment=augment)
    train_len = int(len(dataset) * split)
    split = [train_len, len(dataset) - train_len]
    train_ds, valid_ds = random_split(dataset, split)
    return train_ds, valid_ds

def get_mesh500_1024(root: str | None = None, split: float = 0.8, augment: bool = True) -> tuple[Mesh500, Mesh500]:
    return get_mesh500(root=root, num_points=1024, split=split, augment=augment)

def get_mesh500_4096(root: str | None = None, split: float = 0.8, augment: bool = True) -> tuple[Mesh500, Mesh500]:
    return get_mesh500(root=root, num_points=4096, split=split, augment=augment)