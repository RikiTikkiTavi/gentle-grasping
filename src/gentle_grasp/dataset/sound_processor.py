from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Literal

import torch
import numpy as np
from scipy.io import wavfile
from scipy import signal


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


class Spectrogram2DSoundProcessor(AbstractSoundProcessor):
    def __init__(self, **sound_processing_kwargs):
        self.sound_processing_kwargs = sound_processing_kwargs

    # TODO: Didn't really understand the desired format
    def output_dim(self) -> Literal[1, 2]:
        """
        Returns the output dimension of the sound features.
        1 for single-channel, 2 for stereo.
        """
        return 1

    def read(self, path: Path) -> tuple[torch.Tensor, int]:
        """
        Reads an audio waveform from a file.
        Args:
            path (Path): Path to the audio file.
        Returns:
            tuple[torch.Tensor, int]: A tuple containing the waveform tensor and the sample rate.
        """
        sample_rate, data = wavfile.read(path)

        if data.ndim == 1:
            waveform = torch.from_numpy(data)[None, :]
        else:
            waveform = torch.from_numpy(data.T)
        return waveform, sample_rate

    def transform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Extracts audio features from a waveform.
        Args:
            waveform (torch.Tensor): The audio waveform tensor.
            sample_rate (int): The sample rate of the audio.
        Returns:
            torch.Tensor: A tensor containing the extracted audio features.
        """
        #TODO: Maybe add normalization or other transformations

        specs = []
        for channel in waveform:
            f, t, Sxx = signal.spectrogram(
                channel.numpy(),
                fs=sample_rate,
            )

            Sxx_dB = 10 * np.log10(Sxx + 1e-10)
            specs.append(Sxx_dB)

        spec_tensor = torch.tensor(specs, dtype=torch.float32)
        return spec_tensor

