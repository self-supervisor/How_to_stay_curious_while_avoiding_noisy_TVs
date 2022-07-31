import glob
import numpy as np

def get_files_to_average_over(uncertainty):
    files = glob.glob(f"*noisy_tv_True*curiosity_True*uncertainty_{uncertainty}*action_frequencies*npy")
    return files

def load_list_of_dicts(npy_files):
    list_of_dicts = []
    for a_file in npy_files:
        list_of_dicts.append(np.load(a_file, allow_pickle=True).item())
    return list_of_dicts
        
def average_over_list_of_dicts(list_of_dicts):
    keys = list_of_dicts[0].keys()
    average_dict = {}
    for a_key in keys:
        average_dict[a_key] = 0
    for a_key in keys:
        for a_dict in list_of_dicts:
            average_dict[a_key] += a_dict[a_key][0]
        average_dict[a_key] /= len(list_of_dicts)
    return average_dict

def main():
    uncertainties = [True,False]
    for an_uncertainty in uncertainties:
        npy_files = get_files_to_average_over(an_uncertainty)
        list_of_dicts = load_list_of_dicts(npy_files)
        average_dict = average_over_list_of_dicts(list_of_dicts)
        print(f"Results for uncertainty {an_uncertainty}:")
        print(average_dict)
        

if __name__ == "__main__":
    main()
