from abc import ABC, abstractmethod
from enum import Enum
import os
from pathlib import Path
from typing import Callable, Literal, Sequence
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import torchvision
import copy

from gentle_grasp.dataset.sound_processor import AbstractSoundProcessor


class FileNameTemplate(Enum):
    CAMERA_RGB = "camera_rgb_{step}.png"
    CAMERA_DEPTH = "camera_depth_{step}.png"
    ACTION_HAND = "action{step}_hand.npy"
    ACTION_REGRASP_POSE = "action{step}_regrasp_pose.npy"
    LABELS_SUPERVISED = "labels_supervised.npy"
    TOUCH_MIDDLE = "touch_middle_{step}.png"
    TOUCH_THUMB = "touch_thumb_{step}.png"
    SOUND = "record_s2.wav"


class SampleLoader:
    def __init__(self, sound_processor: AbstractSoundProcessor):
        self.sound_processor = sound_processor

    def _process_tactile_image(self, image_path, reference_path, transform=None):
        image, reference = Image.open(image_path), Image.open(reference_path)
        diff_image = Image.fromarray(np.abs(np.array(image) - np.array(reference)))
        return diff_image

    def _get_path(self, root_dir, step: str | None, template: FileNameTemplate):
        if step is None:
            return os.path.join(root_dir, template.value)
        return os.path.join(root_dir, template.value.format(step=step))

    def __call__(self, data_dir: Path, step: str = "2"):
        # Load visuo images
        camera_rgb = torchvision.datasets.folder.pil_loader(
            self._get_path(
                root_dir=data_dir, step=step, template=FileNameTemplate.CAMERA_RGB
            )
        )
        camera_depth = torchvision.datasets.folder.pil_loader(
            self._get_path(
                root_dir=data_dir, step=step, template=FileNameTemplate.CAMERA_DEPTH
            )
        )

        # Load tactile images
        touch_middle = self._process_tactile_image(
            self._get_path(
                root_dir=data_dir, step=step, template=FileNameTemplate.TOUCH_MIDDLE
            ),
            self._get_path(
                root_dir=data_dir, step="0", template=FileNameTemplate.TOUCH_MIDDLE
            ),
        )
        touch_thumb = self._process_tactile_image(
            self._get_path(
                root_dir=data_dir, step=step, template=FileNameTemplate.TOUCH_THUMB
            ),
            self._get_path(
                root_dir=data_dir, step="0", template=FileNameTemplate.TOUCH_THUMB
            ),
        )

        # Load sound
        sound = self.sound_processor.read(
            Path(
                self._get_path(
                    root_dir=data_dir, step=None, template=FileNameTemplate.SOUND
                )
            )
        )

        labels = torch.tensor(
            np.load(
                self._get_path(
                    root_dir=data_dir,
                    step=None,
                    template=FileNameTemplate.LABELS_SUPERVISED,
                )
            ).astype(np.float32)
        )

        return {
            "camera_rgb": camera_rgb,
            "camera_depth": camera_depth,
            "touch_middle": touch_middle,
            "touch_thumb": touch_thumb,
            "sound": sound,
            "labels": labels,
        }


class StaticSoundAwareLazyDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        transforms: Callable,
        loader: SampleLoader,
        sound_processor: AbstractSoundProcessor,
        label_idx: Sequence[int] = (0,)
    ):
        self.root_dir = root_dir
        self.transform = transforms
        self.samples = []
        self.time_steps = [1, 12, 2]
        self.loader = loader
        self.sound_processor = sound_processor
        self.label_idx = label_idx

        # Collect all valid samples in the root directory
        for p in sorted(root_dir.glob("*")):
            if not p.is_dir():
                continue

            for t in self.time_steps:
                if self.is_valid_sample(p, t):
                    self.samples.append((p, t))

    def is_valid_sample(self, data_dir, idx):
        return (
            os.path.exists(os.path.join(data_dir, f"camera_rgb_{idx}.png"))
            and os.path.exists(os.path.join(data_dir, f"camera_depth_{idx}.png"))
            and os.path.exists(os.path.join(data_dir, f"touch_middle_{idx}.png"))
            and os.path.exists(os.path.join(data_dir, f"touch_thumb_{idx}.png"))
            and os.path.exists(os.path.join(data_dir, "labels_supervised.npy"))
            and os.path.exists(os.path.join(data_dir, f"touch_middle_0.png"))
            and os.path.exists(os.path.join(data_dir, f"touch_thumb_0.png"))
            and os.path.exists(os.path.join(data_dir, FileNameTemplate.SOUND.value))
        )

    def shallow_copy_with_transform(self, t: Callable):
        """Create a shallow copy of the dataset without applying transforms."""
        new_ds = copy.copy(self)
        new_ds.transform = t
        return new_ds

    def __getitem__(self, index: int):

        data_dir, idx = self.samples[index]
        sample = self.loader(data_dir, idx)

        if self.transform:
            camera_rgb = self.transform(sample["camera_rgb"])
            camera_depth = self.transform(sample["camera_depth"])
            touch_middle = self.transform(sample["touch_middle"])
            touch_thumb = self.transform(sample["touch_thumb"])

        sound = self.sound_processor.transform(sample["sound"])

        return {
            "camera_rgb": camera_rgb,
            "camera_depth": camera_depth,
            "touch_middle": touch_middle,
            "touch_thumb": touch_thumb,
            "sound": sound,
            "labels": sample["labels"][list(self.label_idx)],  # Select only specified labels
        }

    def __len__(self):
        return len(self.samples)