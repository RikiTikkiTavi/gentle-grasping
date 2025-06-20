from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, List, Callable
from sympy.polys.fglmtools import monomial_div

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile
from scipy import signal


class AbstractSoundProcessor(ABC):
    """
    Abstract base class for sound processors.

    Defines the interface for reading sound files, extracting features, and applying transformations.
    Utilize torchaudio for transformations.
    """

    @abstractmethod
    def output_dim(self) -> Literal[1, 2]: ...
        # TODO: convert always to mono?


    @abstractmethod
    def read(self, path: Path) -> torch.Tensor:
        """
        Load an audio file from the given path and
        return a waveform tensor of shape (channels, samples).
        """
        ...

    @abstractmethod
    def transform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Applies required transformations to the waveform, including:
            - feature extraction
            - optional augmentations
        """
        ...

    @abstractmethod
    def extraction_transform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Defines the feature extraction pipeline for the waveform.
        """
        ...


class BaseSoundProcessor(AbstractSoundProcessor):
    """
    Base class for common sound processing functionality.

    Subclasses must implement:
        - extraction_transform(): defines the feature extraction pipeline for the waveform.

    Optional augmentation transforms can be provided during initialization.
    All transforms are expected to be Callable.
    """
    def __init__(
            self,
            augmentation_transform: List[Callable] | None = None,
            mono: bool = True,
    ):
        self.sample_rate = 44100
        self.target_shapes: dict | None = None  # Sets Padding and Cropping shapes in format {dimension: shape}
        self.augmentation_transform = augmentation_transform or []
        self.mono = mono

    def output_dim(self) -> Literal[1, 2]: ...

    def read(self, path: Path) -> torch.Tensor:
        """
        Loads an audio file, converts to mono if needed,
        and resamples to the target sample rate.
        """
        waveform, sample_rate = torchaudio.load(path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1 and self.mono:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sample_rate
            )(waveform)

        return waveform

    def _apply_padding(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.target_shapes:
            return tensor

        paddings = [0] * (2 * tensor.ndim)

        for dim, target_len in self.target_shapes.items():
            current_len = tensor.shape[dim]
            if current_len < target_len:
                pad_amount = target_len - current_len
                # F.pad expects padding in reverse order: last dimension first
                paddings[-(2 * dim + 1)] = pad_amount

        padded_tensor = F.pad(tensor, tuple(paddings))
        return padded_tensor

    def _apply_cropping(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.target_shapes:
            return tensor

        slices = [slice(None)] * tensor.ndim
        for dim, target_len in self.target_shapes.items():
            current_len = tensor.shape[dim]
            if current_len > target_len:
                slices[dim] = slice(0, target_len)
        return tensor[tuple(slices)]

    def transform(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.extraction_transform(waveform)
        for aug in self.augmentation_transform:
            features = aug(features)
        return features.squeeze(0)

    @abstractmethod
    def extraction_transform(self, waveform: torch.Tensor) -> torch.Tensor:
        ...


class LogMel2DSoundProcessor(BaseSoundProcessor):
    """
    Sound processor that extracts 2D Log-Mel Spectrogram features from raw audio waveforms.

    This processor converts a mono audio waveform into a time-frequency representation
    using the Mel-scale filter bank, followed by logarithmic compression. The output is a
    2D tensor of shape [n_mels, time_frames], where each value represents the log-scaled
    energy of a mel-frequency band at a given time window.

    Parameters:
        sample_rate (int): Target sample rate for audio processing.
        n_fft (int): Size of the FFT window used for STFT.
        hop_length (int): Number of samples between successive frames (controls time resolution).
        n_mels (int): Number of mel filterbanks (frequency resolution after projection to mel-scale).
        augmentation_transform (List[Callable], optional): List of augmentations
    """

    def __init__(
            self,
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=64,
            augmentation_transform: List[Callable] | None = None,
            mono: bool = True
    ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        super().__init__(augmentation_transform)

# Note: No need for this anymore, as the mel transform is already applied in the transform method(in theory it's the same)
    def extraction_transform(self, waveform: torch.Tensor) -> torch.Tensor:
        features = waveform
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-6)
        # Squeeze [channel, n_mels, time] shape to [n_mels, time]
        features = log_mel_spec.squeeze(0)
        return features
