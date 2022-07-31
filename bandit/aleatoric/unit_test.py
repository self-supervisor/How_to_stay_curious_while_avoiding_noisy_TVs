import numpy as np
import pytest
import random
import matplotlib.pyplot as plt


@pytest.fixture
def bandit_env():
    from bandit_environment import BanditEnvBase
    import os

    env = BanditEnvBase(noise=False)
    return env


@pytest.fixture
def agent():
    from agent import Agent

    agent = Agent(action_space=[0, 1, 2, 3, 4])
    return agent


def test_epsilon(agent, bandit_env):
    import math

    agent.epsilon = 1
    reward = 0
    action = 0
    training_steps = 10000

    for _ in range(training_steps):
        (_, _), reward = bandit_env.step(action)
        print("action", action)
        print("reward", reward)
        action = agent.step(reward)

    for i in agent.action_counts:
        for j in agent.action_counts:
            assert math.isclose(i, j, rel_tol=0.1)


def test_policy(agent):
    agent.epsilon = 0.0
    agent.value = [0, 0, 0, 0, 1]
    action = agent.step(reward=0)
    assert action == 4


def test_step(bandit_env):
    from dataset import generating_function

    for i in range(1000):
        an_action = random.choice(bandit_env.action_space)
        (x, y), reward = bandit_env.step(an_action)
        assert len(x) == len(y)
        assert len(x) == bandit_env.obs_size
        for an_x in x:
            assert an_x > bandit_env.action_region_dict[an_action][0]
            assert an_x < bandit_env.action_region_dict[an_action][1]
        y_points_test = [generating_function(i, noise=False) for i in x]
        assert np.array_equal(np.array(y_points_test), np.array(y))


def test_build_region_dict(bandit_env):
    action_space = [0, 1, 2]
    x_domain = [0, 9]
    region_dict = {0: (0, 3), 1: (3, 6), 2: (6, 9)}
    bandit_env.x_domain = x_domain
    bandit_env.action_space = action_space
    bandit_env.num_of_arms = 3
    assert region_dict == bandit_env.build_action_region_dict()


def test_average_list_of_dictionaries():
    from analyse_results import average_list_of_dictionaries

    dict_2 = {0: 10, 1: 0}
    dict_1 = {0: 1, 1: 2}
        
    np.save("dict_1_ama.npy", dict_1, allow_pickle=True)
    np.save("dict_2_ama.npy", dict_2, allow_pickle=True)
    average = average_list_of_dictionaries(["dict_1_ama.npy", "dict_2_ama.npy"])
    assert average[0] == (10 + 1)/2
    assert average[1] == (2 + 0)/2
