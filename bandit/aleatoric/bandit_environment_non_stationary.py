from bandit_environment import BanditEnvBase
import numpy as np


class BanditEnvNonStationary(BanditEnvBase):
    def __init__(self, noise=False, num_of_arms=5, obs_size=32, x_domain=(-3, 3)):
        super().__init__()
        self.t = 0

    def compute_reward(self, action):
        self.t += 1
        if action == 0:
            reward = 0.1 * (self.t)
        if action == 1:
            reward = 0.2 / (self.t)
        if action == 2:
            reward = 0.3 / (self.t)
        if action == 3:
            reward = 0.4 / (self.t)
        if action == 4:
            reward = 50 / (self.t)
        return reward
