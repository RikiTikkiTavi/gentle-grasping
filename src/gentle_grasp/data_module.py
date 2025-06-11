from pathlib import Path
from sympy import postorder_traversal
import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from pytorch_lightning import LightningDataModule
import torchvision.transforms.v2 as transforms

from gentle_grasp.split_strategy import SplitStrategy
from gentle_grasp.dataset import sound_processor
from gentle_grasp.dataset.static_sound_aware import StaticSoundAwareLazyDataset, SampleLoader
from gentle_grasp.dataset.sound_processor import AbstractSoundProcessor, Dummy1DSoundProcessor



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
        # TODO: Dependency injection

        pre_transforms = [
            transforms.Resize((256, 256)),
        ]

        augmentations = [
            transforms.RandomCrop((224, 224)),
        ]

        post_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]

        sound_processor=Dummy1DSoundProcessor()

        loader = SampleLoader(sound_processor=sound_processor,)

        # Dataset without augmentations to sample validation set
        dataset = StaticSoundAwareLazyDataset(
            root_dir=self.data_path,
            loader=loader,
            sound_processor=sound_processor,
            transforms=transforms.Compose(pre_transforms + post_transforms),
        )
        # Dataset with augmentations to sample training set
        dataset_with_transform = dataset.shallow_copy_with_transform(
            t=transforms.Compose(pre_transforms + augmentations + post_transforms)
        )
        train_idx, val_idx = self.split_strategy.split(list(range(len(dataset))))

        # Get train set from dataset with augmentations
        self.train_dataset = Subset(dataset_with_transform, train_idx)
        # Get validation set from dataset without augmentations
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
