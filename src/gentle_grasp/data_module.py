from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, Subset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold


class GentleGraspDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        val_ratio: float = 0.2,
        cv = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio

        self.cv_enabled  = cv.enabled
        self.n_folds     = cv.n_folds
        self.fold        = cv.fold

    def setup(self, stage=None):
        dataset = torch.load(self.data_path)
        if self.cv_enabled:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            splits = list(kf.split(range(len(dataset))))
            train_idx, val_idx = splits[self.fold]
            self.train_dataset = Subset(dataset, train_idx)
            self.val_dataset = Subset(dataset, val_idx)

        else:
            total_size = len(dataset)
            train_size = int((1 - self.val_ratio) * total_size)
            val_size = total_size - train_size
            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_size, val_size]
            )

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
