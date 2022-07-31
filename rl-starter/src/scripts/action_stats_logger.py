class ActionStatsLogger:
    def __init__(self, number_of_actions):
        self.number_of_actions = number_of_actions
        self.intrinsic_rewards = self.construct_action_stat_dict()
        self.action_frequencies = self.construct_action_stat_dict()
        self.number_of_steps_logged = 0

    def construct_action_stat_dict(self):
        a_dict = {}
        for action in range(self.number_of_actions):
            a_dict[action] = []
        return a_dict

    def add_to_log_dicts(self, action_taken, intrinsic_reward):
        assert len(action_taken) == len(intrinsic_reward)
        for i, _ in enumerate(action_taken):
            self.number_of_steps_logged += 1
            self.add_to_frequency_dict(action_taken[i])
            self.add_to_intrinsic_reward_dict(intrinsic_reward[i], action_taken[i])

    def add_to_frequency_dict(self, action_taken):
        if len(self.action_frequencies[action_taken]) == 0:
            self.action_frequencies[action_taken] = [1]
        else:
            self.action_frequencies[action_taken][0] += 1

    def add_to_intrinsic_reward_dict(self, intrinsic_reward, action_taken):
        if len(self.intrinsic_rewards[action_taken]) == 0:
            self.intrinsic_rewards[action_taken] = [
                [intrinsic_reward, self.number_of_steps_logged]
            ]
        else:
            self.intrinsic_rewards[action_taken].append(
                [intrinsic_reward, self.number_of_steps_logged]
            )

    def save(self, save_name):
        import numpy as np

        np.save(save_name + "_action_frequencies.npy", self.action_frequencies)
        np.save(save_name + "_intrinsic_rewards_per_action.npy", self.intrinsic_rewards)
