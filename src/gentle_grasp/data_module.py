from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from pytorch_lightning import LightningDataModule

from abc import ABC, abstractmethod

from sklearn.model_selection import KFold


class SplitStrategy(ABC):
    @property
    @abstractmethod
    def n_folds(self) -> int: ...

    @abstractmethod
    def split(self, dataset) -> tuple[Dataset, Dataset]:
        """Return train_dataset, val_dataset"""
        pass


class RandomSplit(SplitStrategy):
    def __init__(self, fold, val_size=0.2):
        self.val_size = val_size

    @property
    def n_folds(self) -> int:   
        return 1

    def split(self, dataset):
        val_len = int(len(dataset) * self.val_size)
        train_len = len(dataset) - val_len
        return random_split(dataset, [train_len, val_len])


class CrossValidationSplit(SplitStrategy):
    def __init__(self, fold, n_folds=5):
        self.fold = fold
        self._n_folds = n_folds

    @property
    def n_folds(self) -> int:   
        return self._n_folds

    def split(self, dataset):
        indices = list(
            KFold(n_splits=self.n_folds, shuffle=True, random_state=42).split(
                range(len(dataset))
            )
        )
        train_idx, val_idx = indices[self.fold]
        return Subset(dataset, train_idx), Subset(dataset, val_idx)


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
        dataset = torch.load(self.data_path)
        self.train_dataset, self.val_dataset = self.split_strategy.split(dataset)

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
