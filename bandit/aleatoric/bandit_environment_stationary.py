from bandit_environment import BanditEnvBase
import numpy as np


class BanditEnvStationary(BanditEnvBase):
    def __init__(self, noise=False, num_of_arms=5, obs_size=32, x_domain=(-3, 3)):
        super().__init__()

    def compute_reward(self, action):
        if action == 0:
            reward = np.random.normal(0.1, 0.1)
        if action == 1:
            reward = np.random.normal(0.2, 0.1)
        if action == 2:
            reward = np.random.normal(0.3, 0.1)
        if action == 3:
            reward = np.random.normal(0.4, 0.1)
        if action == 4:
            reward = np.random.normal(0.5, 0.1)
        return reward
