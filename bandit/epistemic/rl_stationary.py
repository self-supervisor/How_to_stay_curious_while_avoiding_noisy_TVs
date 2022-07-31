from bandit_environment_stationary import BanditEnvStationary
from agent import Agent


env = BanditEnvStationary()
agent = Agent(action_space=[0, 1, 2, 3, 4])


reward = 0
action = 0
training_steps = 10000


for _ in range(training_steps):
    (_, _), reward = env.step(action)
    print("action", action)
    print("reward", reward)
    action = agent.step(reward)


print(agent.value)
