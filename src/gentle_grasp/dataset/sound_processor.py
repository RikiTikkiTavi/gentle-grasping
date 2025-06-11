from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import torch


class AbstractSoundProcessor(ABC):
    # Abstract base class for sound processors.
    # Defines the interface for reading and processing sound data.
    # Transform will be used to transform the waveform either in 1D signal or 2D spectrogram.
    # Also probably augmentation transforms will be passed in constructor and applied in transform method.
    # Utilize torchaudio for transformations.

    @abstractmethod
    def output_dim(self) -> Literal[1, 2]: ...

    @abstractmethod
    def read(self, path: Path) -> torch.Tensor: ...

    @abstractmethod
    def transform(self, waveform: torch.Tensor) -> torch.Tensor: ...


class Dummy1DSoundProcessor(AbstractSoundProcessor):
    def __init__(self, **sound_processing_kwargs):
        self.sound_processing_kwargs = sound_processing_kwargs

    def output_dim(self) -> Literal[1, 2]:
        """
        Returns the output dimension of the sound features.
        1 for single-channel, 2 for stereo.
        """
        return 1

    def read(self, path: Path) -> torch.Tensor:
        """
        Reads an audio waveform from a file.
        """
        waveform = torch.zeros(16000)  # mock waveform
        return waveform

    def transform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extracts audio features from a waveform.
        """
        features = torch.zeros(128)  # mock feature vector
        return features
