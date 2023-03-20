import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.cuda
from tqdm import tqdm

import src.latent_controller.controller_utils as c_utils
import src.plot_utils as p_utils
from src.models.end_to_end import EndToEndSystem
from src.timbral_extractor import TimbralFeatureExtractor
from src.wrappers.null_dafx_wrapper import NullDAFXWrapper

parser = ArgumentParser()

parser.add_argument("--checkpoint_dir", type=str,
                    default="/home/kieran/Level5ProjectAudioVAE/src/train_scripts/l5proj_end2end")
parser.add_argument("--data_dir", type=str,
                    default="/home/kieran/Level5ProjectAudioVAE/src/latent_controller/data")
parser.add_argument("--checkpoint_id", type=str, default="th24l5fs")
parser.add_argument("--num_examples", type=int, default=1_000)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--initial_audio_retry", type=int, default=10)

# Parse
args = parser.parse_args()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pl.seed_everything(args.seed)

    y_embeddings = []
    metadata = []

    model = EndToEndSystem.load_from_checkpoint(args.checkpoint_file).to(device)
    model.eval()

    dataset = p_utils.get_audio_dataset(
        dafx=NullDAFXWrapper(),
        num_examples_per_epoch=100,
        length=model.hparams.train_length,
        effect_audio=False,
        batch_size=1,
    )

    feature_extractor = TimbralFeatureExtractor(
        sample_rate=model.hparams.sample_rate,
        features=['brightness', 'depth'],
        default_value=np.NaN
    )

    x_b, x_d = np.NaN, np.NaN
    retries = 0

    while np.isnan(x_b) and np.isnan(x_d):
        x = next(iter(dataset))
        (x_b, x_d), _ = feature_extractor.extract_features(x)
        x_b = x_b.item()
        x_d = x_d.item()

        retries += 1
        if retries > args.initial_audio_retry:
            print(f"Audio retries exceeded: {retries}")
            exit(1)

    x = x.reshape(1, 1, -1).to(device)
    c_utils.save_audio(args.dir, x.cpu().detach().numpy(), "clean")

    dafx = model.dafx

    for i in tqdm(range(args.num_examples)):
        p = dafx.get_random_parameter_settings()
        y = dafx.apply(x, p)
        y = y.reshape(1, 1, -1).to(device)

        z_x, z_y = model.get_audio_embeddings(x, y)

        y_hat, p_hat, _ = model.predict_for_embeddings(x, z_x, z_y)

        (y_hat_b, y_hat_d), _ = feature_extractor.extract_features(y_hat)

        y_hat_b = y_hat_b.item()
        y_hat_d = y_hat_d.item()

        f_name = c_utils.save_audio(args.dir, y_hat.cpu().detach().numpy(), i)

        data_dict = {
            'id': i,
            'audio_file': f_name,
            'x_b': x_b,
            'x_d': x_d,
            'y_hat_b': y_hat_b,
            'y_hat_d': y_hat_d,
            'brightness_diff': y_hat_b - x_b,
            'depth_diff': y_hat_d - x_d
        }

        params = c_utils.param_vector_to_named_dictionary(dafx, p, "p_")
        params_hat = c_utils.param_vector_to_named_dictionary(dafx, p_hat, "p_hat_")

        data_dict.update(params)
        data_dict.update(params_hat)

        y_embeddings.append(z_y.cpu().detach().numpy())
        metadata.append(data_dict)

    embeddings = np.array(y_embeddings).squeeze()
    df = pd.DataFrame(metadata).set_index('id')
    np.save(os.path.join(args.dir, "y_embeddings.npy"), embeddings)
    df.to_csv(os.path.join(args.dir, "metadata.csv"))


if __name__ == "__main__":
    checkpoint_folder = os.path.join(args.checkpoint_dir, args.checkpoint_id + "/checkpoints/")
    args.checkpoint_file = c_utils.get_checkpoint_filename(checkpoint_folder)
    args.dir = os.path.join(args.data_dir, args.checkpoint_id)

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
        print(args.dir, " was created!")

    main(args)
