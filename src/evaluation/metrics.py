import glob
import os
from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm

from src.callbacks.metrics import *
from src.dataset.paired_audio_dataset import PairedAudioDataset
from src.models.end_to_end import EndToEndSystem
from src.plot_utils import dafx_from_name
from src.utils import *


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


def get_results_filename(dafx, args):
    return args.results_dir + f"/{dafx.split()[-1].lower()}_{args.dataset}_{args.num_examples}.csv"


def main(args):
    metrics = {
        "PESQ": PESQ(args.sample_rate),
        "MRSTFT": auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[32, 128, 512, 2048, 8192, 32768],
            hop_sizes=[16, 64, 256, 1024, 4096, 16384],
            win_lengths=[32, 128, 512, 2048, 8192, 32768],
            w_sc=0.0,
            w_phs=0.0,
            w_lin_mag=1.0,
            w_log_mag=1.0,
        ),
        "MSD": MelSpectralDistance(args.sample_rate),
        "SCE": SpectralCentroidError(args.sample_rate),
        "CFE": CrestFactorError(),
        "LUFS": LoudnessError(args.sample_rate),
        "RMS": RMSEnergyError()
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for dafx_name in args.dafx_names:
        print(f"Getting {args.dataset} metrics for: {dafx_name}")
        checkpoint = get_checkpoint_for_effect(dafx_name,
                                               args.checkpoints_dir)

        # load model
        print(f"Loading model from checkpoint: ", checkpoint)
        model = EndToEndSystem.load_from_checkpoint(checkpoint).to(device)
        model.eval()

        # get dataset
        loader = get_dataset(dafx_name,
                             audio_length=model.hparams.train_length * 2,
                             args=args)

        print(f"Predicting audio from reference...")
        outputs = []
        for batch in tqdm(loader):
            x, y = batch
            x, y_ref, y = get_training_reference(x, y)

            y_hat, p, z = model(x.to(device), y=y_ref.to(device))

            outputs.append({
                "y": y.detach().cpu(),
                "y_hat": y_hat.detach().cpu(),
            })

        results = {
            "PESQ": [],
            "MRSTFT": [],
            "MSD": [],
            "SCE": [],
            "CFE": [],
            "LUFS": [],
            "RMS": [],
        }

        print(f"\nCalculating metrics for {dafx_name}...")
        for output in tqdm(outputs):
            for metric_name, metric in metrics.items():
                try:
                    val = metric(output["y_hat"], output["y"])
                    if type(val) == torch.Tensor:
                        val = val.numpy()
                    results[metric_name].append(val)
                except Exception as e:
                    print("Some error occurred: ", e)
                    results[metric_name].append(np.NaN)

        results_filename = get_results_filename(dafx_name, args)
        df = pd.DataFrame(results)
        df.to_csv(results_filename)

        print(f"{dafx_name} metrics saved to: {results_filename}")


parser = ArgumentParser()

parser.add_argument("--dafx_names", nargs="+",
                    default=[
                        # "mda Overdrive",
                        # "mda Delay",
                        # "mda Ambience",
                        # "mda RingMod",
                        # "mda Combo",
                        # "mda Dynamics",
                        # "mda MultiBand",
                        "mda Thru-Zero Flanger",
                        "mda Leslie"
                    ])
parser.add_argument("--dataset", type=str, default="daps")
parser.add_argument("--checkpoints_dir", type=str,
                    default="/home/kieran/Level5ProjectAudioVAE/src/train_scripts/l5proj_end2end")
parser.add_argument("--audio_dir", type=str,
                    default="/home/kieran/Level5ProjectAudioVAE/src/audio")
parser.add_argument("--results_dir", type=str,
                    default="/home/kieran/Level5ProjectAudioVAE/src/evaluation/data/metrics")
parser.add_argument("--num_examples", type=int, default=5_000)
parser.add_argument("--sample_rate", type=int, default=24_000)
parser.add_argument("--seed", type=int, default=123)

# Parse
args = parser.parse_args()

if __name__ == "__main__":
    pl.seed_everything(args.seed)
    args.dataset_input_dirs = [f"{args.dataset}_{args.sample_rate}"]
    main(args)
