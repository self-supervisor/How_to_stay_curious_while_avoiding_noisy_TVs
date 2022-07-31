import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 300)
        self.fc4 = nn.Linear(300, 100)
        self.out_mu = nn.Linear(100, 1)
        self.out_log_sigma_squared = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        mu = self.out_mu(x)
        log_sigma_squared = self.out_log_sigma_squared(x)
        return mu, log_sigma_squared
