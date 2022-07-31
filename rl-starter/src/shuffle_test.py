import pytest
import numpy as np


@pytest.fixture
def noisy_env():
    from utils.env import make_env
    from utils.noisy_tv_wrapper import NoisyTVWrapper

    an_env = make_env(env_key="MiniGrid-MultiRoom-N4-S5-v0", frames_before_reset=0)
    an_env = NoisyTVWrapper(an_env, noisy_tv="True")
    return an_env


def test_add_noisy(noisy_env):
    obs = np.random.randint((7, 7, 3))
    obs_copy = obs.copy()
    obs_dict = {}
    obs_dict["image"] = obs
    assert np.array_equal(obs_dict["image"], obs_copy) == True
    obs_noisy = noisy_env.add_noisy_tv([obs_dict], action=[6])
    assert np.array_equal(obs_noisy, obs_copy) == False
