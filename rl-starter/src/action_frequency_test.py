import pytest 
import glob
import numpy as np
from analyse_action_frequencies import *

def test_get_files_to_average_over():
    uncertainties = [True, False]
    for an_uncertainty in uncertainties:
        all_uncertainty_files = get_files_to_average_over(uncertainty=an_uncertainty)
        assert len(all_uncertainty_files) > 0
        for a_file in all_uncertainty_files:
            assert "curiosity_True" in a_file
            assert "uncertainty_" + str(an_uncertainty) in a_file
            an_uncertainty ^= True # toggle a boolean
            assert "uncertainty_" + str(an_uncertainty) not in a_file
            an_uncertainty ^= True # toggle back
            
        assert len(set(all_uncertainty_files)) == len(all_uncertainty_files)


def test_load_list_of_dicts():
    import math
    
    uncertainties = [True, False]
    for an_uncertainty in uncertainties:
        npy_files = get_files_to_average_over(uncertainties)
        list_of_dicts = load_list_of_dicts(npy_files)
        assert len(list_of_dicts) == len(npy_files)
        for a_dict in list_of_dicts:
            assert list(a_dict.keys()) == [0, 1, 2, 3, 4, 5, 6]
            values = [*a_dict.values()]
            values = [item for sublist in values for item in sublist]
            assert math.isclose(sum(values), 50000, abs_tol=1000)


def test_average_over_list_of_dicts():
    dict_1 = {}
    dict_1["0"] = [10]
    dict_1["1"] = [40]
    dict_2 = {}
    dict_2["0"] = [30]
    dict_2["1"] = [0]
    fake_list_of_dicts = [dict_1, dict_2]
    average_dict = average_over_list_of_dicts(fake_list_of_dicts)
    assert average_dict["0"] == 20
    assert average_dict["1"] == 20
