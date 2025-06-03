import os
from pathlib import Path
from typing import Callable
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import torchvision
import copy


def _process_tactile_image(image_path, reference_path, transform=None):
    image, reference = Image.open(image_path), Image.open(reference_path)
    diff_image = Image.fromarray(np.abs(np.array(image) - np.array(reference)))
    return diff_image

def _sample_loader(data_dir, idx):
    # Load visuo-tactile images
    camera_rgb = torchvision.datasets.folder.pil_loader(os.path.join(data_dir, f"camera_rgb_{idx}.png"))
    camera_depth = torchvision.datasets.folder.pil_loader(os.path.join(data_dir, f"camera_depth_{idx}.png"))
    touch_middle = _process_tactile_image(
        os.path.join(data_dir, f"touch_middle_{idx}.png"),
        os.path.join(data_dir, "touch_middle_0.png"),
    )
    touch_thumb = _process_tactile_image(
        os.path.join(data_dir, f"touch_thumb_{idx}.png"),
        os.path.join(data_dir, "touch_thumb_0.png"),
    )
    # Load actions and labels
    hand_action = (
        torch.tensor(
            np.load(os.path.join(data_dir, "action1_hand.npy")).astype(np.float32)
        )
        if os.path.exists(os.path.join(data_dir, "action1_hand.npy"))
        else None
    )
    relpose_action = (
        torch.tensor(
            np.load(os.path.join(data_dir, "action1_regrasp_pose.npy")).astype(
                np.float32
            )
        )
        if idx != 2
        and os.path.exists(os.path.join(data_dir, "action1_regrasp_pose.npy"))
        else torch.zeros(4, dtype=torch.float32)
    )
    labels = (
        torch.tensor(
            np.load(os.path.join(data_dir, "labels_supervised.npy")).astype(
                np.float32
            )
        )
        if os.path.exists(os.path.join(data_dir, "labels_supervised.npy"))
        else None
    )

    return (
        (camera_rgb, camera_depth),
        (touch_middle, touch_thumb),
        (hand_action, relpose_action),
        labels,
    )

class LazyDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        transforms: Callable,
        loader=_sample_loader
    ):
        self.root_dir = root_dir
        self.transform = transforms
        self.samples = []
        self.time_steps = [1, 12, 2]
        self.loader = loader

        def is_valid_sample(data_dir, idx):
            return (
                os.path.exists(os.path.join(data_dir, f"camera_rgb_{idx}.png"))
                and os.path.exists(os.path.join(data_dir, f"camera_depth_{idx}.png"))
                and os.path.exists(os.path.join(data_dir, f"touch_middle_{idx}.png"))
                and os.path.exists(os.path.join(data_dir, f"touch_thumb_{idx}.png"))
                and os.path.exists(os.path.join(data_dir, "action1_hand.npy"))
                and os.path.exists(os.path.join(data_dir, "action1_regrasp_pose.npy"))
                and os.path.exists(os.path.join(data_dir, "labels_supervised.npy"))
                and os.path.exists(os.path.join(data_dir, f"touch_middle_0.png"))
                and os.path.exists(os.path.join(data_dir, f"touch_thumb_0.png"))
            )

        for p in sorted(root_dir.glob("*")):
            if not p.is_dir():
                continue

            for t in self.time_steps:
                if is_valid_sample(p, t):
                    self.samples.append((p, t))

    def shallow_copy_with_transform(self, t: Callable):
        """Create a shallow copy of the dataset without applying transforms."""
        new_ds = copy.copy(self)
        new_ds.transform = t
        return new_ds

    def __getitem__(self, index: int):
        
        data_dir, idx = self.samples[index]
        sample = self.loader(data_dir, idx)

        camera_rgb, camera_depth = sample[0]
        touch_middle, touch_thumb = sample[1]
        hand_action, relpose_action = sample[2]
        labels = sample[3]

        if self.transform:
            camera_rgb = self.transform(camera_rgb)
            camera_depth = self.transform(camera_depth)
            touch_middle = self.transform(touch_middle)
            touch_thumb = self.transform(touch_thumb)

        return (
            (camera_rgb, camera_depth),
            (touch_middle, touch_thumb),
            (hand_action, relpose_action),
            labels,
        )

    def __len__(self):
        return len(self.samples)
