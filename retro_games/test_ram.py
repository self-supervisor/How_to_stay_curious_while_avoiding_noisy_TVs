import gym
import numpy as np

env = gym.make("BankHeistNoFrameskip-v4")
obs = env.reset()
obs, reward, done, info = env.step(1)
