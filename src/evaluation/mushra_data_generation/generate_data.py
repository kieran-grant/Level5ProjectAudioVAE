import os
from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
import torch
from scipy.io import wavfile
from tqdm import tqdm

from src.evaluation.evaluation_utils import get_checkpoint_for_effect, get_dataset, apply_pedalboard_effect
from src.models.end_to_end import EndToEndSystem
from src.plot_utils import dafx_from_name
from src.utils import get_training_reference, peak_normalise

parser = ArgumentParser()

parser.add_argument("--num_examples", type=int, default=3)
parser.add_argument("--dafx_names", nargs="+", default=[
    "mda Overdrive",
    "mda MultiBand",
    "mda Delay",
    "mda Thru-Zero Flanger",
    "mda Ambience",
    ])

parser.add_argument("--dataset", type=str, default="daps")
parser.add_argument("--checkpoints_dir", type=str,
                    default="/home/kieran/Level5ProjectAudioVAE/src/train_scripts/l5proj_end2end")
parser.add_argument("--audio_dir", type=str,
                    default="/home/kieran/Level5ProjectAudioVAE/src/audio")
parser.add_argument("--results_dir", type=str,
                    default="/home/kieran/Level5ProjectAudioVAE/src/evaluation/data/mushra_audio")
parser.add_argument("--style_transfer_dir", type=str,
                    default="/home/kieran/DeepAFx-ST/")
parser.add_argument("--sample_rate", type=int, default=24_000)
parser.add_argument("--seed", type=int, default=1234)

args = parser.parse_args()


def save_audio(data_dir, signal, filename, sample_rate=24_000):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(data_dir, " was created!")

    if type(signal) == torch.Tensor:
        signal = signal.detach().cpu().numpy()

    f_name_path = f"{data_dir}/{filename}"
    wavfile.write(f_name_path, sample_rate, signal.squeeze())
    return f_name_path


def generate_st_command(x_path, y_ref_path, fname, args):
    out_dir = f"{args.results_dir}/st_y_hat/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(out_dir, " was created!")

    return "python " \
           "scripts/process.py " \
           f"-i {x_path} " \
           f"-r {y_ref_path} " \
           "-c checkpoints/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch\=362-step\=1210241-val-jamendo-autodiff.ckpt " \
           f"-out_dir {out_dir} " \
           f"-out_f_name {fname} "


if __name__ == "__main__":
    pl.seed_everything(args.seed)
    args.dataset_input_dirs = [f"{args.dataset}_{args.sample_rate}"]

    for dafx_name in args.dafx_names:
        audio_settings = []
        st_commands = ["#! /bin/bash"]

        print(f"Generating data for {dafx_name}...")
        checkpoint_id = get_checkpoint_for_effect(dafx_name, args.checkpoints_dir)
        print(f"Getting {args.dataset} metrics for: {dafx_name}")
        checkpoint = get_checkpoint_for_effect(dafx_name,
                                               args.checkpoints_dir)

        # load model
        print(f"Loading model from checkpoint: ", checkpoint)
        model = EndToEndSystem.load_from_checkpoint(checkpoint)
        model.eval()

        dataset = get_dataset(
            dafx_name="clean",
            audio_length=model.hparams.train_length * 2,
            args=args
        )

        dafx = dafx_from_name(dafx_name)

        dafx_name = dafx_name.split()[-1].lower()
        for i, batch in tqdm(enumerate(dataset)):
            x, y = batch

            # create effected audio (REFERENCE)
            y, settings = apply_pedalboard_effect(effect_name=dafx_name, audio=y, sr=args.sample_rate)
            y = peak_normalise(y)

            x, y_ref, y = get_training_reference(x, y)

            # apply fixed audio settings (ANCHOR)
            rand_p = dafx.get_random_parameter_settings()
            rand_y_hat = dafx.apply(x, rand_p)
            rand_y_hat = peak_normalise(rand_y_hat)

            # predict with end-to-end model
            e2e_y_hat, p, z = model(x, y=y_ref)
            e2e_y_hat = peak_normalise(e2e_y_hat)

            fname = f"{dafx_name}{i}.wav"

            x_file = save_audio(f"{args.results_dir}/x/", signal=x, filename=fname)
            y_file = save_audio(f"{args.results_dir}/y/", signal=y, filename=fname)
            y_ref_file = save_audio(f"{args.results_dir}/y_ref/", signal=y_ref, filename=fname)
            rand_y_file = save_audio(f"{args.results_dir}/rand_y_hat/", signal=rand_y_hat, filename=fname)
            e2e_y_file = save_audio(f"{args.results_dir}/e2e_y_hat/", signal=e2e_y_hat, filename=fname)

            # if compression, also get style transfer approx
            if dafx_name.lower() == "multiband":
                cmd = generate_st_command(x_file, y_ref_file, args=args, fname=fname)
                st_commands.append(cmd)

            settings["id"] = i

            # Add parameter settings to setting list
            for j in range(p.size()[-1]):
                settings[f"p_{dafx.idx_to_param_map[j]}"] = p.squeeze()[j].item()

            audio_settings.append(settings)

        df = pd.DataFrame(audio_settings)
        df.to_csv(f"{args.results_dir}/{dafx_name}.csv")

        if len(st_commands) > 1:
            with open(f"{args.results_dir}/{dafx_name}_script.sh", "w") as f:
                for line in st_commands:
                    f.write(f"{line}\n")
