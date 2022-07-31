import glob
import pandas as pd
import numpy as np


def average_list_of_dictionaries(paths_to_dictionaries):

    list_of_dictionaries = [
        np.load(i, allow_pickle=True).item() for i in paths_to_dictionaries
    ]
    keys = list_of_dictionaries[0].keys()
    mean_dict = {}
    for a_key in keys:
        mean_dict[a_key] = 0
    for a_dict in list_of_dictionaries:
        for a_key in keys:
            mean_dict[a_key] += a_dict[a_key]
    for a_key in keys:
        mean_dict[a_key] /= len(list_of_dictionaries)
    return mean_dict


results = glob.glob("*.npy")
ama_list = []
mse_list = []

for a_result in results:
    if "mse" in a_result:
        mse_list.append(a_result)
    elif "ama" in a_result:
        ama_list.append(a_result)
    else:
        raise ValueError("file name incorrect, must contain MSE or AMA tag")


print("AMA average", average_list_of_dictionaries(ama_list))
print("MSE average", average_list_of_dictionaries(mse_list))
