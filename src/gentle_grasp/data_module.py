from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from pytorch_lightning import LightningDataModule
import torchvision.transforms.v2 as transforms

from abc import ABC, abstractmethod

from sklearn.model_selection import KFold, train_test_split

from gentle_grasp.dataset import LazyDataset


class SplitStrategy(ABC):
    @property
    @abstractmethod
    def n_folds(self) -> int: ...

    @abstractmethod
    def split(self, indices) -> tuple[list[int], list[int]]:
        """Return train_dataset, val_dataset"""
        pass


class RandomSplit(SplitStrategy):
    def __init__(self, fold, val_size=0.2):
        self.val_size = val_size

    @property
    def n_folds(self) -> int:
        return 1

    def split(self, indices):
        train_indices, val_indices = train_test_split(
            indices, test_size=self.val_size, random_state=42, shuffle=True
        )
        return train_indices, val_indices


class CrossValidationSplit(SplitStrategy):
    def __init__(self, fold, n_folds=5):
        self.fold = fold
        self._n_folds = n_folds

    @property
    def n_folds(self) -> int:
        return self._n_folds

    def split(self, indices):
        l2indices = list(
            KFold(n_splits=self.n_folds, shuffle=True, random_state=42).split(indices)
        )
        train_l2_idx, val_l2_idx = l2indices[self.fold]
        return indices[train_l2_idx], indices[val_l2_idx]


class GentleGraspDataModule(LightningDataModule):
    def __init__(
        self,
        split_strategy: SplitStrategy,
        data_path: Path,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_strategy = split_strategy

    def setup(self, stage=None):
        dataset = LazyDataset(
            self.data_path,
            transforms=transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            ),
        )
        dataset_with_transform = dataset.shallow_copy_with_transform(
            t=transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            ),
        )
        train_idx, val_idx = self.split_strategy.split(list(range(len(dataset))))

        self.train_dataset = Subset(dataset_with_transform, train_idx)
        self.val_dataset = Subset(dataset, val_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
