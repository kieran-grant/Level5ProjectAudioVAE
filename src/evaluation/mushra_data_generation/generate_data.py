from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
import torch
from scipy.io import wavfile
from tqdm import tqdm

from src.evaluation.evaluation_utils import get_checkpoint_for_effect, get_dataset, apply_pedalboard_effect
from src.models.end_to_end import EndToEndSystem
from src.plot_utils import dafx_from_name
from src.utils import get_training_reference

parser = ArgumentParser()

parser.add_argument("--num_examples", type=int, default=5)
parser.add_argument("--dafx_names", nargs="+", default=["mda Overdrive", "mda MultiBand"])
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
parser.add_argument("--seed", type=int, default=123)

args = parser.parse_args()


def save_audio(data_dir, signal, filename, sample_rate=24_000):
    if type(signal) == torch.Tensor:
        signal = signal.detach().cpu().numpy()

    f_name_path = f"{data_dir}/{filename}"
    wavfile.write(f_name_path, sample_rate, signal.squeeze())
    return f_name_path


def generate_st_command(x_path, y_ref_path):
    return "python " \
           "scripts/process.py " \
           f"-i {x_path} " \
           f"-r {y_ref_path} " \
           "-c checkpoints/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch\=362-step\=1210241-val-jamendo-autodiff.ckpt"


if __name__ == "__main__":
    pl.seed_everything(args.seed)
    args.dataset_input_dirs = [f"{args.dataset}_{args.sample_rate}"]

    audio_settings = []
    st_commands = ["#! /bin/bash"]

    for dafx_name in args.dafx_names:
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
            x, y_ref, y = get_training_reference(x, y)

            # apply random audio settings (ANCHOR)
            rand_y_hat = dafx.process_audio_with_random_settings(x)

            # predict with end-to-end model
            e2e_y_hat, p, z = model(x, y=y_ref)

            x_file = save_audio(args.results_dir, signal=x, filename=f"x_{i}.wav")
            y_file = save_audio(args.results_dir, signal=y, filename=f"y_{i}.wav")
            y_ref_file = save_audio(args.results_dir, signal=y_ref, filename=f"y_ref_{i}.wav")
            rand_y_file = save_audio(args.results_dir, signal=rand_y_hat, filename=f"rand_y_hat_{i}.wav")
            e2e_y_file = save_audio(args.results_dir, signal=e2e_y_hat, filename=f"e2d_y_hat_{i}.wav")

            # if compression, also get style transfer approx
            if dafx_name.lower() == "multiband":
                cmd = generate_st_command(x_file, y_ref_file)
                st_commands.append(cmd)

            settings["id"] = i
            audio_settings.append(settings)

        df = pd.DataFrame(audio_settings)
        df.to_csv(f"{args.results_dir}/{dafx_name}.csv")

        if len(st_commands) > 1:
            with open(f"{args.results_dir}/comp_script.sh", "w") as f:
                for line in st_commands:
                    f.write(f"{line}\n")
