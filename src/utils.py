import torch
import numpy as np


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
