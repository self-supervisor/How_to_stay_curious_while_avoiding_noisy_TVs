import argparse
import glob
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--seeds", help="list of seeds to aggregate over")
args = parser.parse_args()

all_csv_files = glob.glob("*csv")
seeds = args.seeds.split(",")

experiment_strings = [word[:-6] for word in all_csv_files]
experiment_strings = list(set(experiment_strings))
print(experiment_strings)

for experiment_string in experiment_strings:
    experiments = list(filter(lambda k: experiment_string in k, all_csv_files))
    N_rows, N_cols = pd.read_csv(experiments[0]).shape
    results = np.zeros((N_rows, N_cols))
    for csv_file in experiments:
        results += pd.read_csv(csv_file).values
    results /= len(experiments)
    np.save(experiment_string + "mean.npy", results)
