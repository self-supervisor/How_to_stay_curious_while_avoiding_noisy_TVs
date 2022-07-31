#!/usr/bin/env python
# coding: utf-8
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

# get_ipython().run_line_magic("matplotlib", "inline")
plt.rc("font", family="serif")
plt.rc("xtick", labelsize="large")
plt.rc("ytick", labelsize="large")

mndata = MNIST("data")
x_train_data, y_train_data = mndata.load_training()
x_test_data, y_test_data = mndata.load_testing()

training_steps = 50000
checkpoint_loss = 1000
device = torch.device("cuda:1")
print("device:", device)


class NoisyMnistEnv:
    def __init__(
        self, split, input_number_min, input_number_max, batch_size=32, seed=0
    ):
        self.seed = seed
        self.split = split
        if self.split == "train":
            self.x, self.y = x_train_data, y_train_data
        elif self.split == "test":
            self.x, self.y = x_test_data, y_test_data
        self.batch_size = batch_size
        self.input_number_min = input_number_min
        self.input_number_max = input_number_max

    def step(self):
        x_arr = np.zeros((self.batch_size, 28 * 28))
        y_arr = np.zeros((self.batch_size, 28 * 28))

        for i in range(self.batch_size):
            input_number = np.random.randint(
                self.input_number_min, self.input_number_max
            )
            if input_number == 0:
                output_number = 0
            if input_number == 1:
                output_number = np.random.randint(2, 10)
            input_data = self.get_random_sample_of_number(input_number)
            if input_number == 1:
                output_data = self.get_random_sample_of_number(output_number)
            elif input_number == 0:
                output_data = input_data
            x_arr[i] = np.array(input_data)
            y_arr[i] = np.array(output_data)
        return x_arr, y_arr

    def get_random_sample_of_number(self, number):
        random_num = np.random.randint(0, len(self.y) - 1)
        if self.y[random_num] == number:
            return self.x[random_num]
        else:
            return self.get_random_sample_of_number(number)


# from here https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_1 = nn.Linear(28 * 28, 784)
        self.linear_2 = nn.Linear(784, 784)
        self.linear_3 = nn.Linear(784, 784)
        self.linear_4 = nn.Linear(784 * 2, 28 * 28)

    def forward(self, input_x):
        x = F.relu(self.linear_1(input_x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        x = torch.cat((input_x, x), dim=1)
        x = self.linear_4(x)
        return x


class AleatoricNet(nn.Module):
    def __init__(self):
        super(AleatoricNet, self).__init__()
        self.linear_1 = nn.Linear(28 * 28, 784)
        self.linear_2 = nn.Linear(784, 784)
        self.linear_3_mu = nn.Linear(784, 784)
        self.linear_4_mu = nn.Linear(784 * 2, 28 * 28)
        self.linear_3_sigma = nn.Linear(784, 784)
        self.linear_4_sigma = nn.Linear(784, 28 * 28)

    def forward(self, input_x):
        x = F.relu(self.linear_1(input_x))
        x = F.relu(self.linear_2(x))
        mu = F.relu(self.linear_3_mu(x))
        mu = torch.cat((input_x, mu), dim=1)
        mu = self.linear_4_mu(mu)
        log_sigma = F.relu(self.linear_3_sigma(x))
        log_sigma = self.linear_4_sigma(log_sigma)
        return mu, log_sigma


class NoisyMNISTExperimentRun:
    def __init__(
        self,
        repeats,
        training_steps,
        checkpoint_loss,
        lr,
        model,
        mnist_env_train,
        mnist_env_test_zeros,
        mnist_env_test_ones,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.training_steps = training_steps
        self.checkpoint_loss = checkpoint_loss
        self.repeats = repeats
        self.model = model
        self.device = device
        self.lr = lr
        self.env_train = mnist_env_train
        self.env_test_zeros = mnist_env_test_zeros
        self.env_test_ones = mnist_env_test_ones
        self.device = device
        self.reset_model()
        self.reset_loss_buffers()

    def run_experiment(self):
        for repeat in range(self.repeats):
            self.reset_model()
            self.reset_loss_buffers()
            for update in tqdm(range(int(self.training_steps))):
                self.train_step(update)
                self.eval_step("ones", update)
                self.eval_step("zeros", update)
            self.save_evaluation_performance(repeat)

    def save_evaluation_performance(self, repeat):
        np.save(
            "mse_loss_list_deterministic_repeat_" + str(repeat) + ".npy",
            self.loss_list_0,
        )
        np.save(
            "mse_loss_list_stochastic_repeat_" + str(repeat) + ".npy", self.loss_list_1
        )

    def preprocess_batch(self, data, target):
        data /= 255
        target /= 255
        return data, target

    def compute_loss_and_reward(self, prediction, target):
        loss = F.mse_loss(prediction, target)
        reward = loss
        return loss, reward

    def get_batch(self, env):
        data, target = env.step()
        data, target = self.preprocess_batch(data, target)
        data = torch.from_numpy(data).float().to(self.device)
        target = torch.from_numpy(target).float().to(self.device)
        return data, target

    def train_step(self, update):
        update += 1
        self.model.train()
        data, target = self.get_batch(self.env_train)
        self.opt.zero_grad()
        output = self.model(data)
        loss, reward = self.compute_loss_and_reward(output, target)
        loss.backward()
        self.opt.step()
        self.loss_buffer.append(reward)
        if update % self.checkpoint_loss == 0:
            self.loss_list.append(
                torch.mean(torch.stack(self.loss_buffer)).detach().cpu().numpy()
            )
            self.loss_buffer = []

    def eval_step(self, ones_or_zeros, update):
        update += 1
        self.model.eval()
        assert ones_or_zeros in ["ones", "zeros"]
        if ones_or_zeros == "ones":
            env = self.env_test_zeros
        elif ones_or_zeros == "zeros":
            env = self.env_test_ones
        data, target = self.get_batch(env)
        output = self.model(data)
        loss, reward = self.compute_loss_and_reward(output, target)
        if ones_or_zeros == "ones":
            self.loss_buffer_1.append(reward)
            if update % self.checkpoint_loss == 0:
                self.loss_list_1.append(
                    torch.mean(torch.stack(self.loss_buffer_1)).detach().cpu().numpy()
                )
                self.loss_buffer_1 = []
        elif ones_or_zeros == "zeros":
            self.loss_buffer_0.append(reward)
            if update % self.checkpoint_loss == 0:
                self.loss_list_0.append(
                    torch.mean(torch.stack(self.loss_buffer_0)).detach().cpu().numpy()
                )
                self.loss_buffer_0 = []

    def reset_model(self):
        self.model = self.model.to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

    def reset_loss_buffers(self):
        self.loss_list = []
        self.loss_buffer = []
        self.loss_list_0 = []
        self.loss_list_1 = []
        self.loss_buffer_0 = []
        self.loss_buffer_1 = []


class NoisyMNISTExperimentRunAMA(NoisyMNISTExperimentRun):
    def __init__(
        self,
        repeats,
        training_steps,
        checkpoint_loss,
        lr,
        model,
        mnist_env_train,
        mnist_env_test_zeros,
        mnist_env_test_ones,
        device,
    ):
        NoisyMNISTExperimentRun.__init__(
            self,
            repeats,
            training_steps,
            checkpoint_loss,
            lr,
            model,
            mnist_env_train,
            mnist_env_test_zeros,
            mnist_env_test_ones,
            device,
        )

    def compute_loss_and_reward(self, prediction, target):
        mu, log_sigma = prediction
        mse = F.mse_loss(mu, target, reduction="none")
        loss = torch.mean(torch.exp(-log_sigma) * mse + log_sigma)
        reward = torch.mean(mse - torch.exp(log_sigma))
        return loss, reward

    def save_evaluation_performance(self, repeat):
        np.save(
            "aleatoric_loss_list_deterministic_repeat_" + str(repeat) + ".npy",
            self.loss_list_0,
        )
        np.save(
            "aleatoric_loss_list_stochastic_repeat_" + str(repeat) + ".npy",
            self.loss_list_1,
        )


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aleatoric_model = AleatoricNet()
    mndata = MNIST("data")
    x_train_data, y_train_data = mndata.load_training()
    x_test_data, y_test_data = mndata.load_testing()

    mnist_env_train = NoisyMnistEnv("train", 0, 2)
    mnist_env_test_zeros = NoisyMnistEnv("test", 0, 1)
    mnist_env_test_ones = NoisyMnistEnv("test", 1, 2)

    repeats = 1
    training_steps = 500
    checkpoint_loss = 10
    aleatoric_lr = 0.001

    config = {
        "repeats": repeats,
        "training_steps": training_steps,
        "checkpoint_loss": checkpoint_loss,
        "aleatoric_lr": aleatoric_lr,
    }
