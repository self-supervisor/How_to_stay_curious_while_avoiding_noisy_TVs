#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np

def get_experiments():
    all_csvs = glob.glob("*.csv")
    filtered_csvs = []
    for i, value in enumerate(all_csvs):
        if "curiosity_False" in value:
            continue
        filtered_csvs.append(value)
    assert all_csvs != filtered_csvs
    csvs_without_suffix = [item[:-6] for item in filtered_csvs]
    experiments = list(set(csvs_without_suffix))
    return experiments, filtered_csvs

def get_all_files_of_same_type(experiments):
    dict_holding_files_and_run_types = {}
    for a_run_type in experiments:
        all_files_of_run_type = []
        for csv_file in filtered_csvs:
            if a_run_type in csv_file:
                all_files_of_run_type.append(csv_file)
        dict_holding_files_and_run_types[a_run_type] = all_files_of_run_type
    return dict_holding_files_and_run_types

def average_over_run_type(run_csvs):
    import csv

    data_list = []
    
    for a_csv in run_csvs:
        with open(a_csv, 'r') as f:
            reader = csv.reader(f)
            data_as_list = list(reader)
        f.close()
        data_list.append(data_as_list)
            
    data_list = np.array(data_list, dtype=np.float64)
    data_list = np.mean(data_list, axis=0)
    return data_list

def average_over_multiple_run_types(files_of_same_type, list_of_run_types):
    values = []
    for run_type in list_of_run_types:
        value = average_over_run_type(files_of_same_type[run_type])
        values.append(value)
    values = np.array(values)
    return np.mean(values, axis=0)

def get_best_hyperparam(average_performance_with_and_without_tv):
    max_val = 0
    for row in average_performance_with_and_without_tv:
        if row[2] >= max_val:
            best_row = row
            max_val = row[2]
    print("icm lr: " + str(best_row[0]))
    print("reward_weighting: " + str(best_row[1]))
    print("novel_states_visited: " + str(best_row[2]))
    return best_row


experiments, filtered_csvs = get_experiments()
print("Without Uncertainty best hyperparams....")
files_of_same_type = get_all_files_of_same_type(experiments)
average_performance_with_and_without_tv = average_over_multiple_run_types(files_of_same_type, ["frames_8_noisy_tv_False_curiosity_True_uncertainty_False_random", "frames_8_noisy_tv_False_curiosity_True_uncertainty_False_random"])
get_best_hyperparam(average_performance_with_and_without_tv)
print("With Uncertainty best hyperparams...")
files_of_same_type = get_all_files_of_same_type(experiments)
average_performance_with_and_without_tv = average_over_multiple_run_types(files_of_same_type, ["frames_8_noisy_tv_False_curiosity_True_uncertainty_True_random", "frames_8_noisy_tv_False_curiosity_True_uncertainty_True_random"])
get_best_hyperparam(average_performance_with_and_without_tv)
