import torch


def read_sound(sound_path: str) -> torch.Tensor:
    """
    Reads an audio waveform from a file.
    """
    waveform = torch.zeros(16000)
    return waveform


def process_sound(waveform: torch.Tensor) -> torch.Tensor:
    """
    Extracts audio features from a waveform.
    """
    features = torch.zeros(128)  # mock feature vector
    return features
