from pathlib import Path
import numpy as np
from sympy import postorder_traversal
import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from pytorch_lightning import LightningDataModule
import torchvision.transforms.v2 as transforms

from abc import ABC, abstractmethod

from sklearn.model_selection import KFold, train_test_split


class SplitStrategy(ABC):
    @property
    @abstractmethod
    def n_folds(self) -> int: ...

    @abstractmethod
    def split(self, indices) -> tuple[list[int], list[int]]:
        """Return train_dataset, val_dataset"""
        pass


class RandomSplit(SplitStrategy):
    def __init__(self, val_size=0.2, seed: int = 42, fold=None):
        self.val_size = val_size
        self._seed = seed

    @property
    def n_folds(self) -> int:
        return 1

    def split(self, indices):
        train_indices, val_indices = train_test_split(
            indices, test_size=self.val_size, random_state=self._seed, shuffle=True
        )
        return train_indices, val_indices


class CrossValidationSplit(SplitStrategy):
    def __init__(self, fold, n_folds=5, seed: int = 42):
        self.fold = fold
        self._n_folds = n_folds
        self._seed = seed

    @property
    def n_folds(self) -> int:
        return self._n_folds

    def split(self, indices):
        l2indices: list[tuple[np.ndarray, np.ndarray]] = list(
            KFold(n_splits=self.n_folds, shuffle=True, random_state=self._seed).split(
                indices
            )
        )
        train_l2_idx, val_l2_idx = l2indices[self.fold]
        return [indices[i] for i in train_l2_idx], [indices[i] for i in val_l2_idx]
