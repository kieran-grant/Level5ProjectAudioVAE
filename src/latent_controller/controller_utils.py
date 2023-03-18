import glob
import os

import torch
from scipy.io import wavfile


def get_checkpoint_filename(checkpoint_folder):
    list_of_files = glob.glob(checkpoint_folder + "/*.ckpt")
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def save_audio(data_dir, signal, idx, sample_rate=24_000):
    audio_dir = data_dir + "/audio/"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print(audio_dir, " was created!")
    f_name = f"audio_{idx}.wav"
    wavfile.write(audio_dir + f_name, sample_rate, signal)
    return f_name


def param_vector_to_named_dictionary(dafx, vec, prefix=""):
    vec = vec.squeeze()

    if type(vec) == torch.Tensor:
        vec = vec.cpu().detach().numpy()

    assert (len(vec.shape) == 1)

    out = {}

    for i in range(len(vec)):
        param_name = dafx.idx_to_param_map[i]
        out[f"{prefix}{param_name}"] = vec[i]

    return out