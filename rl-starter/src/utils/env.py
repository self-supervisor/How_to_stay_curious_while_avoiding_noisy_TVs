import gym
import gym_minigrid


def make_env(env_key, frames_before_reset, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    env.reset()
    # env = DeterministicResetWrapper(env, frames_before_reset, seed)
    return env
