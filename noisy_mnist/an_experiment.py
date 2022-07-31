from tqdm import tqdm
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from noisy_mnist_aleatoric_uncertainty_for_poster import (
    Net,
    NoisyMNISTExperimentRun,
    AleatoricNet,
    NoisyMNISTExperimentRunAMA,
    NoisyMNISTExperimentRunLearningProgress,
)
from noisy_mnist_aleatoric_uncertainty_for_poster import NoisyMnistEnv


def run_experiment(reward_function):
    mndata = MNIST("data")
    x_train_data, y_train_data = mndata.load_training()
    x_test_data, y_test_data = mndata.load_testing()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    mnist_env_train = NoisyMnistEnv("train", 0, 2)
    mnist_env_test_zeros = NoisyMnistEnv("test", 0, 1)
    mnist_env_test_ones = NoisyMnistEnv("test", 1, 2)

    repeats = 1
    training_steps = 5000
    checkpoint_loss = 10
    mse_lr = 0.001
    aleatoric_lr = 0.0001
    if reward_function == "mse":
        lr = 0.001
    elif reward_function == "ama":
        lr = 0.0001
    else:
        raise ValueError("reward function must be ama or mse")

    config = {
        "training steps": training_steps,
        "checkpoint loss": checkpoint_loss,
        "learning rate": lr,
        "reward function": reward_function,
    }
    wandb.config.update(config)

    if reward_function == "mse":
        model = Net()
        experiment = NoisyMNISTExperimentRun(
            repeats=repeats,
            training_steps=training_steps,
            checkpoint_loss=checkpoint_loss,
            lr=lr,
            model=model,
            mnist_env_train=mnist_env_train,
            mnist_env_test_zeros=mnist_env_test_zeros,
            mnist_env_test_ones=mnist_env_test_ones,
            device=device,
        )
    elif reward_function == "ama":
        model = AleatoricNet()
        experiment = NoisyMNISTExperimentRunAMA(
            repeats=repeats,
            training_steps=training_steps,
            checkpoint_loss=checkpoint_loss,
            lr=lr,
            model=model,
            mnist_env_train=mnist_env_train,
            mnist_env_test_zeros=mnist_env_test_zeros,
            mnist_env_test_ones=mnist_env_test_ones,
            device=device,
        )
    experiment.run_experiment()


if __name__ == "__main__":
    wandb.init(project="Noisy MNIST")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_fn", help="ama or mse", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args.reward_fn)
