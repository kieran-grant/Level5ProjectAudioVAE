from argparse import ArgumentParser

import pandas as pd
import umap
from tqdm import tqdm

from src.latent_controller.controller_utils import *
from src.plot_utils import get_colour


def get_supervised_feature_ids(df, n, tolerance=1_000):
    duplicates = []
    iterations = 1

    def retrieve_ids(current_df, idx):
        return current_df['id'].values[idx]

    while True:
        current_df = df[~df['id'].isin(duplicates)]

        top_bright_idx = get_top_n_indices(current_df, 'brightness_diff', n)
        top_bright = retrieve_ids(current_df, top_bright_idx)

        bottom_bright_idx = get_bottom_n_indices(current_df, 'brightness_diff', n)
        bottom_bright = retrieve_ids(current_df, bottom_bright_idx)

        top_depth_idx = get_top_n_indices(current_df, 'depth_diff', n)
        top_depth = retrieve_ids(current_df, top_depth_idx)

        bottom_depth_idx = get_bottom_n_indices(current_df, 'depth_diff', n)
        bottom_depth = retrieve_ids(current_df, bottom_depth_idx)

        intersection = common_elements(top_bright, bottom_bright, top_depth, bottom_depth)

        if len(intersection) == 0:
            print(f"\nFound top {n} features in {iterations} iteration(s)!")
            return top_bright, bottom_bright, top_depth, bottom_depth

        duplicates.extend(intersection)

        iterations += 1

        if iterations >= tolerance:
            print(f"Iterations surpassed tolerance of {tolerance}, no IDs returned")
            return None, None, None, None


def set_class_labels(df, bright_top, bright_bottom, deep_top, deep_bottom):
    # give labels to columns
    df['class'] = -1

    df.loc[df['id'].isin(bright_top), 'class'] = 1
    df.loc[df['id'].isin(bright_bottom), 'class'] = 2
    df.loc[df['id'].isin(deep_top), 'class'] = 3
    df.loc[df['id'].isin(deep_bottom), 'class'] = 4


parser = ArgumentParser()

parser.add_argument("--data_dir", type=str,
                    default="/home/kieran/Level5ProjectAudioVAE/src/latent_controller/data")
parser.add_argument("--checkpoint_id", type=str, default="fboro0y2")
parser.add_argument("--n_supervised", nargs="+", default=[0, 5, 10, 25, 100])
parser.add_argument("--n_neighbors", type=int, default=15)
parser.add_argument("--min_dist", type=float, default=.1)
parser.add_argument("--metric", type=str, default="euclidean")

# Parse
args = parser.parse_args()

if __name__ == "__main__":
    args.dir = args.data_dir + "/" + args.checkpoint_id

    y_emb = np.load(os.path.join(args.dir, 'y_embeddings.npy'))
    metadata = pd.read_csv(os.path.join(args.dir, "metadata.csv"))
    embeddings = dict()

    for N in tqdm(args.n_supervised):
        reducer = umap.UMAP(n_neighbors=args.n_neighbors,
                            min_dist=args.min_dist,
                            metric=args.metric)

        if N == 0:
            emb = reducer.fit_transform(y_emb)

        else:
            b_t, b_b, d_t, d_b = get_supervised_feature_ids(metadata, N)
            set_class_labels(metadata, b_t, b_b, d_t, d_b)
            masked_target = metadata['class'].values

            emb = reducer.fit_transform(y_emb, y=masked_target)

        color_emb = (emb - emb.min(0)) / emb.ptp(0)
        colours = np.array([get_colour(e[0], e[1]) for e in color_emb], dtype=str)
        colours = np.expand_dims(colours, axis=1)

        complete_emb = np.hstack([emb, colours])

        embeddings[f'n={N}'] = complete_emb

    for k, v in embeddings.items():
        metadata[f'x_emb_{k}'] = v[:, 0]
        metadata[f'y_emb_{k}'] = v[:, 1]
        metadata[f'colour_{k}'] = v[:, 2]

    metadata.to_csv(os.path.join(args.dir, "full_data.csv"))
