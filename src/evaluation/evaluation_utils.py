import glob
import os

import numpy as np
import torch
from pedalboard import HighShelfFilter, LowShelfFilter, Compressor, Distortion, Delay, Reverb
from pedalboard.pedalboard import Pedalboard

from src.dataset.paired_audio_dataset import PairedAudioDataset
from src.plot_utils import dafx_from_name
from src.utils import peak_normalise, effect_to_end_to_end_checkpoint_id


def apply_pedalboard_effect(effect_name: str, audio: torch.Tensor, sr: int = 24_000):
    implementations = {
        "combo": apply_pedalboard_distortion,
        "overdrive": apply_pedalboard_distortion,
        "delay": apply_pedalboard_delay,
        "multiband": apply_pedalboard_compression,
        "ambience": apply_pedalboard_reverb,
    }

    name = effect_name.lower()
    if name in implementations.keys():
        device = audio.device
        audio = audio.cpu().numpy()
        y, settings = implementations[name](audio.squeeze(), sr=sr)
        y_tensor = torch.Tensor(y.reshape(1, 1, -1)).to(device)
        y_tensor = peak_normalise(y_tensor)
        return y_tensor, settings
    else:
        raise NotImplementedError("Effect must be one of: ", list(implementations.keys()))


def apply_pedalboard_delay(audio, sr=24_000):
    board = Pedalboard([])
    settings = {}

    delay_secs = np.random.randint(5, 10) / 10
    feedback = np.random.randint(5, 10) / 10

    settings["delay_secs"] = delay_secs
    settings["delay_feedback"] = feedback
    board.append(Delay(delay_seconds=delay_secs, feedback=feedback))

    effected_audio = board(audio, sr)

    return effected_audio, settings


def apply_pedalboard_distortion(audio, sr=24_000):
    board = Pedalboard([])
    settings = {}

    drive_db = np.random.randint(25, 40)

    settings["dist_drive_db"] = drive_db
    board.append(Distortion(drive_db=drive_db))

    effected_audio = board(audio, sr)

    return effected_audio, settings


def apply_pedalboard_compression(audio, sr=24_000):
    board = Pedalboard([])
    settings = {}

    # Add high shelf boost/cut with probability 0.5
    if np.random.rand() < 0.5:
        gain = np.random.randint(-5, 5)
        hz = np.random.randint(500, 2_000)

        settings["high_shelf_cutoff_hz"] = hz
        settings["high_shelf_gain_db"] = gain

        board.append(HighShelfFilter(gain_db=gain, cutoff_frequency_hz=hz))

    # Add low shelf boost/cut with probability 0.5
    if np.random.rand() < 0.5:
        gain = np.random.randint(-5, 5)
        hz = np.random.randint(20, 500)

        settings["low_shelf_cutoff_hz"] = hz
        settings["low_shelf_gain_db"] = gain

        board.append(LowShelfFilter(gain_db=gain, cutoff_frequency_hz=hz))

    # Always add some amount of compression
    ratio = np.random.randint(1, 25)
    threshold = np.random.randint(-20, -1)

    settings["comp_ratio"] = ratio
    settings["comp_threshold"] = threshold

    board.append(Compressor(threshold_db=threshold, ratio=ratio))

    effected_audio = board(audio, sr)

    return effected_audio, settings


def apply_pedalboard_reverb(audio, sr=24_000):
    board = Pedalboard([])
    settings = {}

    room_size = np.random.randint(2, 10) / 10

    settings["reverb_room_size"] = room_size
    board.append(Reverb(room_size=room_size))

    effected_audio = board(audio, sr)

    return effected_audio, settings


def get_val_checkpoint_filename(checkpoint_folder):
    list_of_files = glob.glob(checkpoint_folder + "/*.ckpt")
    val_file = [fl for fl in list_of_files if "val" in fl]
    latest_file = max(val_file, key=os.path.getctime)
    return latest_file


def get_checkpoint_for_effect(effect_name, checkpoints_dir):
    checkpoint_id = effect_to_end_to_end_checkpoint_id(effect_name)
    checkpoint_id_dir = os.path.join(checkpoints_dir, checkpoint_id + "/checkpoints/")
    checkpoint_file = get_val_checkpoint_filename(checkpoint_id_dir)
    return checkpoint_file


def get_dataset(dafx_name, audio_length, args):
    dafx = dafx_from_name(dafx_name)

    dataset = PairedAudioDataset(
        dafx=dafx,
        audio_dir=args.audio_dir,
        subset="train",
        train_frac=0.95,
        input_dirs=args.dataset_input_dirs,
        num_examples_per_epoch=args.num_examples,
        augmentations={},
        length=audio_length,
        effect_input=False,
        effect_output=True,
        random_effect_threshold=0.,
        dummy_setting=False
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=2,
        batch_size=1,
        timeout=6000,
        generator=g,
    )



