from __future__ import print_function

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import rcParams
from matplotlib.pyplot import figure
from mnist import MNIST
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm

# from here https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder
class AleatoricNet(nn.Module):
    def __init__(self):
        super(AleatoricNet, self).__init__()
        self.sequential_feat_extract = nn.Sequential(
            nn.Linear(28 * 28, 784), nn.ReLU(), nn.Linear(784, 784)
        )
        self.sequential_mu = nn.Sequential(
            nn.Linear(784, 784), nn.ReLU(), nn.Linear(784 * 2, 28 * 28)
        )
        self.sequential_log_sigma = nn.Sequential(
            nn.Linear(784, 784), nn.ReLU(), nn.Linear(784, 28 * 28)
        )

    def forward(self, input_x):
        mu = self.sequential_feat_extract(input_x)
        mu = self.sequential_mu(input_x)
        log_sigma = self.sequential_feat_extract(input_x)
        log_sigma = self.sequential_log_sigma(input_x)
        return mu, log_sigma


model = AleatoricNet()
torch.save(model, "aleatoric_model.h5")
