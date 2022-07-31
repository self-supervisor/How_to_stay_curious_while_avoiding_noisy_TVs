import numpy as np


class Agent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_counts = [0] * len(action_space)
        self.value = [0] * len(action_space)
        self.action = 0
        self.epsilon = 0.1
        self.rewards = [[] for _ in action_space]
        self.alpha = 1  # 0.98

    def step(self, reward):
        self.rewards[self.action].append(reward)
        self.action_counts[self.action] += 1
        self.update_values(reward)
        self.action = self.policy()
        return self.action

    def policy(self):
        if np.random.uniform(0, 1) > self.epsilon:
            return np.random.choice(
                np.flatnonzero(np.array(self.value) == np.array(self.value).max())
            )
        else:
            return np.random.choice(self.action_space)

    def update_values(self, reward):
        self.value[self.action] += self.alpha * (reward - self.value[self.action])
