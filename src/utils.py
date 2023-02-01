import numpy as np
import torch

from torchaudio.transforms import MFCC


def conform_length(x: torch.Tensor, length: int):
    """Crop or pad input on last dim to match `length`."""
    if x.shape[-1] < length:
        padsize = length - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, padsize))
    elif x.shape[-1] > length:
        x = x[..., :length]

    return x


def linear_fade(
        x: torch.Tensor,
        fade_ms: float = 50.0,
        sample_rate: float = 22050,
):
    """Apply fade in and fade out to last dim."""
    fade_samples = int(fade_ms * 1e-3 * sample_rate)

    fade_in = torch.linspace(0.0, 1.0, steps=fade_samples)
    fade_out = torch.linspace(1.0, 0.0, steps=fade_samples)

    # fade in
    x[..., :fade_samples] *= fade_in

    # fade out
    x[..., -fade_samples:] *= fade_out

    return x


def peak_normalise(signal: torch.Tensor):
    # peak normalize to -12 dBFS
    signal /= signal.abs().max()
    signal *= 10 ** (-12.0 / 20)  # with min 3 dBFS headroom
    return signal


def is_silent(signal: torch.Tensor):
    """
    Returns True if signal is silent
    """

    signal = signal.detach().cpu().numpy()
    mean = np.square(signal).mean()

    return mean < 1e-5


def is_noise(original: torch.Tensor,
             effected: torch.Tensor,
             sample_rate: int,
             cosine_similarity_threshold: float = 0.75):
    """
    Returns True if difference between original and effected signal is too extreme.
    """
    mfcc = MFCC(sample_rate=sample_rate,
                n_mfcc=30,
                melkwargs={"n_mels": 60})

    a = mfcc(original)
    b = mfcc(effected)

    similarity = torch.nn.functional.cosine_similarity(a, b).mean()
    # print(similarity)
    return similarity < cosine_similarity_threshold


def split_dataset(file_list, subset, train_frac):
    """Given a list of files, split into train/val/test sets.

    Args:
        file_list (list): List of audio files.
        subset (str): One of "train", "val", or "test".
        train_frac (float): Fraction of the dataset to use for training.

    Returns:
        file_list (list): List of audio files corresponding to subset.
    """
    assert 0.1 < train_frac < 1.0

    total_num_examples = len(file_list)

    train_num_examples = int(total_num_examples * train_frac)
    val_num_examples = int(total_num_examples * (1 - train_frac) / 2)
    test_num_examples = total_num_examples - (train_num_examples + val_num_examples)

    if train_num_examples < 0:
        raise ValueError(
            f"No examples in training set. Try increasing train_frac: {train_frac}."
        )
    elif val_num_examples < 0:
        raise ValueError(
            f"No examples in validation set. Try decreasing train_frac: {train_frac}."
        )
    elif test_num_examples < 0:
        raise ValueError(
            f"No examples in test set. Try decreasing train_frac: {train_frac}."
        )

    if subset == "train":
        start_idx = 0
        stop_idx = train_num_examples
    elif subset == "val":
        start_idx = train_num_examples
        stop_idx = start_idx + val_num_examples
    elif subset == "test":
        start_idx = train_num_examples + val_num_examples
        stop_idx = start_idx + test_num_examples + 1
    else:
        raise ValueError("Invalid subset: {subset}.")

    return file_list[start_idx:stop_idx]


def audio_to_spectrogram(signal: torch.Tensor,
                         n_fft: int = 4096,
                         hop_length: int = 2048,
                         window_size: int = 4096):
    window = torch.nn.Parameter(torch.hann_window(window_size))

    # compute spectrogram of waveform
    X = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )

    X_db = torch.pow(X.abs() + 1e-8, 0.3)
    X_db_norm = X_db

    # standardize (0, 1) 0.322970 0.278452
    X_db_norm -= 0.322970
    X_db_norm /= 0.278452
    X_db_norm = X_db_norm.permute(0, 2, 1)

    return X_db_norm.detach()
