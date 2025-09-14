from torch.utils.data import Dataset
from torch import Tensor

from datasets import load_dataset

import numpy as np
import torch


class Ellipsoid1024(Dataset):
    def __init__(self, root: str, split: str = "train"):
        super().__init__()

        self._root = root
        self._split = split
        self._dataset = load_dataset("kohido/ellipsoid_1024pts", cache_dir=self._root)[self._split]

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tensor:
        points = self._dataset[index]['points']
        points: np.ndarray = np.array(points)
        points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
        points: Tensor = torch.from_numpy(points)
        points = points.float()
        points = points.permute(1, 0)
        return points


def get_ellipsoid1024(root: str | None = None) -> tuple[Ellipsoid1024, Ellipsoid1024]:
    assert root is not None, "Please provide the path to the Ellipsoid1024 dataset root directory"

    train_ds = Ellipsoid1024(root=root, split="train")
    valid_ds = Ellipsoid1024(root=root, split="val")
    return train_ds, valid_ds

def make_grid_ellipsoid1024(points: Tensor) -> Tensor:
    pass