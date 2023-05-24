import pandas as pd

import glob
import re

RESULTS_DIR = "/home/kieran/Level5ProjectAudioVAE/src/evaluation/data/metrics"
DATASETS = ["daps", "vctk", "musdb18"]

files = glob.glob(f"{RESULTS_DIR}/*.csv")

for dataset in DATASETS:
    dataset_files = [fl for fl in files if dataset in fl]
    dataset_files = [fl for fl in dataset_files if "baseline" in fl]
    dataframes = []
    for file_name in dataset_files:
        # Extract the name of the effect using regex
        match = re.search(r'/(?P<effect>[^/]+)_' + dataset + '_', file_name)

        try:
            effect_name = match.group('effect')
        except:
            continue

        df = pd.read_csv(file_name, index_col=0)
        df['dafx'] = effect_name
        df.columns = [col.lower() for col in df.columns]
        cols = list(df.columns)
        cols = [cols[-1]] + cols[:-1]
        df = df[cols]
        dataframes.append(df)

    df = pd.concat(dataframes)
    df.to_csv(f"{RESULTS_DIR}/{dataset}_baseline.csv")
