from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm

from src.callbacks.metrics import *
from src.evaluation.evaluation_utils import  get_dataset

from src.utils import *


def get_results_filename(dafx, args):
    return args.results_dir + f"/{dafx.split()[-1].lower()}_{args.dataset}_{args.num_examples}_baseline.csv"


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
    }

    for dafx_name in args.dafx_names:
        print(f"Getting {args.dataset} metrics for: {dafx_name}")

        loader = get_dataset(dafx_name,
                             audio_length=131072,
                             args=args)

        print(f"Predicting audio from reference...")
        outputs = []
        for batch in tqdm(loader):
            x, y = batch

            outputs.append({
                "y": y.detach().cpu(),
                "y_hat": x.detach().cpu(),  # predict no effect
            })

        results = {
            "PESQ": [],
            "MRSTFT": [],

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
                        "mda Overdrive",
                        "mda Delay",
                        "mda Ambience",
                        "mda RingMod",
                        "mda Combo",
                        "mda Dynamics",
                        "mda MultiBand",
                        "mda Thru-Zero Flanger",
                        "mda Leslie"
                    ])
parser.add_argument("--dataset", type=str, default="vctk")
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
