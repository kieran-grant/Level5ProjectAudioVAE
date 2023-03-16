import math

import torch
from pedalboard.pedalboard import load_plugin

from src.dataset.audio_dataset import AudioDataset
from src.wrappers.dafx_wrapper import DAFXWrapper
from src.wrappers.null_dafx_wrapper import NullDAFXWrapper


def dafx_from_name(dafx_name,
                   sample_rate=24_000,
                   dafx_file="/home/kieran/Level5ProjectAudioVAE/src/dafx/mda.vst3"):
    if dafx_name.lower() == "clean":
        return NullDAFXWrapper()
    dafx = load_plugin(dafx_file, plugin_name=dafx_name)
    return DAFXWrapper(dafx=dafx, sample_rate=sample_rate)


def get_audio_dataset(dafx,
                      num_examples_per_epoch,
                      length=131_072,
                      effect_audio=True,
                      dummy_setting=False,
                      audio_dir="/home/kieran/Level5ProjectAudioVAE/src/audio",
                      input_dirs=["vctk_24000"],
                      num_workers=8,
                      batch_size=1,
                      ):
    dataset = AudioDataset(
        dafx=dafx,
        audio_dir=audio_dir,
        subset="train",
        input_dirs=input_dirs,
        num_examples_per_epoch=num_examples_per_epoch,
        augmentations={},
        length=length,
        effect_audio=effect_audio,
        dummy_setting=dummy_setting
    )

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        timeout=6000,
    )


def get_colour(x, y):
    assert (0 <= x <= 1)
    assert (0 <= y <= 1)

    # calculate the red, green, and blue values based on x and y coordinates
    red = int(x * 255)
    green = int(y * 255)
    blue = int(0.5 * 255)

    def clamp(color):
        return max(0, min(color, 255))

    hex = "#{0:02x}{1:02x}{2:02x}".format(clamp(red), clamp(green), clamp(blue))

    # return the RGBA value as hex
    return hex


def get_subplot_dimensions(N, max_columns=None):
    if max_columns is None or N <= max_columns:
        # If max_columns is not specified or N is less than or equal to max_columns,
        # then the number of columns required is equal to N.
        # In this case, the number of rows required is 1.
        return 1, N

    # If N is greater than max_columns, we need to calculate the number of rows required.
    rows = math.ceil(N / max_columns)
    cols = max_columns

    return rows, cols


def calculate_upper_triangular_entries(n):
    return int((n * (n - 1)) / 2)


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]
