import numpy as np


class BanditEnvBase:
    def __init__(
        self,
        noise=False,
        num_of_arms=14,
        obs_size=32,
        x_domain=((-np.pi), (5 * np.pi / 2)),
    ):
        self.num_of_arms = num_of_arms
        self.action_space = range(self.num_of_arms)
        self.t = 0
        self.obs_size = obs_size
        self.noise = noise
        self.x_domain = x_domain
        self.action_region_dict = self.build_action_region_dict()

    def compute_reward(self, action):
        return np.random.uniform(0, 1)

    def step(self, action):
        low, high = self.action_region_dict[action]
        x_points = np.random.uniform(low, high, size=self.obs_size)
        y_points = self.gen_obs(x_points)
        reward = self.compute_reward(action)
        return (x_points, y_points), reward

    def gen_obs(self, x_points):
        from dataset import generating_function

        y = []
        for an_x in x_points:
            y.append(generating_function(an_x))

        return y

    def build_action_region_dict(self):
        region_size = (self.x_domain[1] - self.x_domain[0]) / self.num_of_arms
        action_region_dict = {}
        for arm in range(self.num_of_arms):

            action_region_dict[arm] = (
                self.x_domain[0] + arm * region_size,
                self.x_domain[0] + (arm + 1) * region_size,
            )

        return action_region_dict
