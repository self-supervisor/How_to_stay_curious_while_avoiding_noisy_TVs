import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_mean_and_std_dev(csv_paths, quantity):
    runs = []
    for csv_path in csv_paths:
        num_frames_mean = extract_num_frames_mean_from_csv(csv_path, quantity)
        num_frames_mean = np.array([float(i) for i in num_frames_mean])
        runs.append(num_frames_mean)
    min_length = min([len(i) for i in runs])
    runs = [run[:min_length] for run in runs]
    runs = np.array(runs)
    return np.mean(runs, axis=0), np.std(runs, axis=0)


def plot_mean_and_uncertainty(mean, std, label, num_of_points, multiply_factor):
    if "curiosity True uncertainty True" in label:
        color = "red"
        label = "AMA Curiosity"
    elif "curiosity False uncertainty False" in label:
        color = "blue"
        label = "No Reward A2C"
    elif "curiosity True uncertainty False" in label:
        color = "yellow"
        label = "MSE Curiosity"
    else:
        color = "purple"
        label = "random actions"
    plt.plot(
        np.array(range(num_of_points)) * multiply_factor, mean, label=label, color=color
    )
    plt.fill_between(
        np.array(range(num_of_points)) * multiply_factor,
        mean - std,
        mean + std,
        alpha=0.2,
        color=color,
    )


def extract_num_frames_mean_from_csv(csv_path, quantity):
    df = pd.read_csv(csv_path)
    df = df.drop(df[df["update"] == "update"].index)
    num_frames_mean = df[quantity]
    return np.array(num_frames_mean)


def get_label_from_path(path_string):
    # format storage/noisy_tv_True_curiosity_True_uncertainty_True_random_seed_29_coefficient_0.0005
    label = path_string.split("/")[1].split("_random")[0].replace("_", " ")
    return label


def plot(title, path_strings, quantity):
    from matplotlib.pyplot import figure

    figure(num=None, figsize=(8, 6), dpi=80, facecolor="w", edgecolor="k")
    ### formatting ###
    plt.rcParams["axes.formatter.limits"] = [-5, 5]
    plt.rc("font", family="serif")
    plt.rc("xtick", labelsize="medium")
    plt.rc("ytick")
    ### calculations ###
    for path_string_list in path_strings:
        path_string_list = [s + "/log.csv" for s in path_string_list]
        mean, std = get_mean_and_std_dev(path_string_list, quantity)
        frames = path_string_list[0].split("/")[1][0:9]
        label = get_label_from_path(path_string_list[0])
        plot_mean_and_uncertainty(mean, std, label, len(mean), 2.5e5 / len(mean))

    plt.xlabel("Total Frames Elapsed", fontsize=15)
    plt.ylabel(quantity.replace("_", " "), fontsize=15)
    plot_title = title + " " + quantity
    plt.title(plot_title.replace("_", " "))
    plt.legend(loc="best")
    plt.savefig(plot_title + "_" + frames + ".png")


def ignore_empty_lists(a_list_of_lists):
    new_list = []
    for an_item in a_list_of_lists:
        if len(an_item) == 0:
            continue
        new_list.append(an_item)
    return new_list


def main(args):
    quantities_to_plot = ["intrinsic_rewards", "novel_states_visited", "uncertainties"]
    all_strings = glob.glob("storage/*80*")
    print(all_strings)
    for quantity in quantities_to_plot:
        Curious_True_Noisy_True_Uncertain_True = []
        Curious_False_Noisy_True_Uncertain_False = []
        Curious_False_Noisy_False_Uncertain_False = []
        Curious_True_Noisy_False_Uncertain_True = []
        Curious_True_Noisy_False_Uncertain_False = []
        Curious_True_Noisy_True_Uncertain_False = []
        random_Noisy_True = []
        random_Noisy_False = []

        # format: noisy_tv_True_curiosity_True_uncertainty_True_random_seed_29_coefficient_0.0005
        for string in all_strings:
            if "curiosity_True" in string:
                if "noisy_tv_True" in string:
                    if "uncertainty_True" in string:
                        if "random_action" not in string:
                            Curious_True_Noisy_True_Uncertain_True.append(string)

            if "curiosity_False" in string:
                if "noisy_tv_True" in string:
                    if "uncertainty_False" in string:
                        if "random_action" not in string:
                            Curious_False_Noisy_True_Uncertain_False.append(string)

            if "curiosity_False" in string:
                if "noisy_tv_False" in string:
                    if "uncertainty_False" in string:
                        if "random_action" not in string:
                            Curious_False_Noisy_False_Uncertain_False.append(string)

            if "curiosity_True" in string:
                if "noisy_tv_False" in string:
                    if "uncertainty_True" in string:
                        if "random_action" not in string:
                            Curious_True_Noisy_False_Uncertain_True.append(string)

            if "curiosity_True" in string:
                if "noisy_tv_False" in string:
                    if "uncertainty_False" in string:
                        if "random_action" not in string:
                            Curious_True_Noisy_False_Uncertain_False.append(string)

            if "curiosity_True" in string:
                if "noisy_tv_True" in string:
                    if "uncertainty_False" in string:
                        if "random_action" not in string:
                            Curious_True_Noisy_True_Uncertain_False.append(string)
            
            if "random_action" in string:
                if "noisy_tv_False" in string:
                    random_Noisy_False.append(string)
                elif "noisy_tv_True" in string:
                    random_Noisy_True.append(string)

        path_strings_noisy_tv = [
            Curious_True_Noisy_True_Uncertain_True,
            Curious_False_Noisy_True_Uncertain_False,
            Curious_True_Noisy_True_Uncertain_False,
            random_Noisy_True,
        ]
        path_strings_no_noisy = [
            Curious_False_Noisy_False_Uncertain_False,
            Curious_True_Noisy_False_Uncertain_True,
            Curious_True_Noisy_False_Uncertain_False,
            random_Noisy_False,
        ]
    
        path_strings_noisy_tv = ignore_empty_lists(path_strings_noisy_tv)
        path_strings_no_noisy = ignore_empty_lists(path_strings_no_noisy)

        if len(path_strings_noisy_tv) > 0:
            plot("With Noisy TV ", path_strings_noisy_tv, quantity)
        if len(path_strings_no_noisy) > 0:
            plot("Without Noisy TV ", path_strings_no_noisy, quantity)
        #plot("With Noisy TV " + args.environment + args.reward_weighting + args.normalise_rewards + args.icm_lr, path_strings_noisy_tv, quantity)
        #plot("Without Noisy TV " + args.environment + args.reward_weighting + args.normalise_rewards + args.icm_lr, path_strings_no_noisy, quantity)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str)
    #parser.add_argument("--reward_weighting", type=str)
    #parser.add_argument("--normalise_rewards", type=str)
    #parser.add_argument("--icm_lr", type=str)
    args = parser.parse_args()
    main(args)
