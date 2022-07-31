import glob
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers import Monitor
from wrappers import (
    MontezumaInfoWrapper,
    make_mario_env,
    make_robo_pong,
    make_robo_hockey,
    make_multi_pong,
    AddRandomStateToInfo,
    MaxAndSkipEnv,
    ProcessFrame84,
    ExtraTimeLimit,
    StickyActionEnv,
    NoisyTVEnvWrapper,
    ActionLoggingWrapper,
)


def get_car_mask(frames, car_color=np.array([223, 183, 85])):
    mask = np.zeros(shape=frames[0].shape)
    for a_frame in frames:
        for i in range(a_frame.shape[0]):
            for j in range(a_frame.shape[1]):
                if np.array_equal(a_frame[i][j], car_color):
                    mask[i][j] += 1
    return mask


env = gym.make("BankHeistNoFrameskip-v4")
env = gym.wrappers.Monitor(env, "./video/", force=True)
env._max_episode_steps = 4000 * 4
env = MaxAndSkipEnv(env, skip=4)
env = ProcessFrame84(env, crop=False)
env = FrameStack(env, 4)
env = ExtraTimeLimit(env, 4000)
env = AddRandomStateToInfo(env)

obs = env.reset()

for _ in range(100):
    obs, reward, done, info = env.step(env.action_space.sample())
    import pdb

    pdb.set_trace()
