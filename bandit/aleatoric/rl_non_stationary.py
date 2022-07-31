import matplotlib.pyplot as plt
from matplotlib import animation
from bandit_environment_non_stationary import BanditEnvNonStationary
from agent import Agent
from copy import copy

env = BanditEnvNonStationary()
agent = Agent(action_space=[0, 1, 2, 3, 4])


reward = 0
action = 0
training_steps = 100

values = [[] for _ in range(training_steps)]


for i in range(training_steps):
    (_, _), reward = env.step(action)
    print("action", action)
    print("reward", reward)
    action = agent.step(reward)
    print(agent.value)
    values[i] = copy(agent.value)
    print(values)

for i, value_distribution in enumerate(values):
    print("value distribution", value_distribution)
    plt.bar(range(len(value_distribution)), value_distribution)
    plt.savefig(f"bar_charts/chart_{i:0>8}.png")
    plt.close()

import subprocess

process = subprocess.Popen(
    f"convert -delay 30 -loop 0 bar_charts/*.png bar_chart.gif",
    shell=True,
    stdout=subprocess.PIPE,
)
process.wait()
