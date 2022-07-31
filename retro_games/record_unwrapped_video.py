import gym
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
import numpy as np
import time


class BankHeistWrapperRecorder(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.reset_car_mask()

    def reset_car_mask(self):
        self.car_mask = {
            "104, 72, 98": np.zeros((250, 160)),
            "66, 158, 130": np.zeros((250, 160)),
            "252, 252, 84": np.zeros((250, 160)),
            "236, 200, 96": np.zeros((250, 160)),
        }

    def close(self):
        self.plot_car_mask()
        self.reset_car_mask()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.update_car_mask(obs)
        if done == True:
            self.plot_car_mask()
            self.reset_car_mask()
        return obs, reward, done, info

    def update_car_mask(self, a_frame, car_color=np.array([223, 183, 85])):
        maze_count = 0
        for a_key in self.car_mask.keys():
            if np.array([int(i) for i in a_key.split(",")]) in a_frame:
                maze_count += 1
                self.car_mask[a_key] += np.sum(
                    np.where(np.array(a_frame) == car_color, [1, 1, 1], [0, 0, 0]),
                    axis=2,
                )
        assert maze_count < 2  # should only be in one maze at a time

    def plot_car_mask(self):
        import wandb
        import matplotlib.pyplot as plt
        import time

        unix_timestamp = time.time()
        image = np.zeros((250, 160 * 4))
        for i, a_key in enumerate(self.car_mask.keys()):
            image[:, i * 160 : (i + 1) * 160] = self.car_mask[a_key]

        # plt.imshow(np.log(image))
        # plt.colorbar()
        # wandb.log({f"car_mask_plot_log_scale_{unix_timestamp}": plt})
        # wandb.log({"novel states": np.sum(np.where(np.log(image) == -np.inf, 0, 1))})
        # plt.close()
        np.save(f"episode/car_mask_{unix_timestamp}.npy", image)

