# https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream
import torch
import numpy as np


class OnlineVariance(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, device, iterable=None, ddof=1):
        self.device = device
        self.ddof, self.n, self.mean, self.M2 = (
            torch.tensor(ddof, device=self.device),
            torch.tensor(0, device=self.device),
            torch.tensor(0.0, device=self.device),
            torch.tensor(1.0, device=self.device),
        )
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)
        normalised_reward = (datum - self.mean) / (self.std + 1e-7)
        return normalised_reward

    def include_tensor(self, tensor):
        normalised_reward = []
        for _, val in enumerate(tensor):
            normalised_reward.append(self.include(val))
        normalised_reward = torch.stack(normalised_reward)
        # normalised_reward[normalised_reward != normalised_reward] = 0
        return normalised_reward

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return torch.sqrt(self.variance)
