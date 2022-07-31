from dataset import generating_function
import matplotlib.pyplot as plt
from bandit_environment import BanditEnvBase
import numpy as np
import torch
import torch.optim as optim
from model import Net
from dataset import get_data
import torch.nn.functional as F
from copy import deepcopy
from agent import Agent
import wandb

CUDA_DEVICE = "cuda:1"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
noisy_actions = [8, 9, 10, 11]
stable_actions = [2, 3, 4, 5]


def build_model():
    network = Net()
    network.to(CUDA_DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    network.train()
    return network, optimizer


def update_model(network, optimizer, x, y, gradient_update, reward_function):
    optimizer.zero_grad()
    network_copy = deepcopy(network)
    mu, log_sigma_squared = network(x.reshape(32, 1))
    mse = F.mse_loss(mu, y.reshape(32, 1), reduction="none")
    loss = 0.5 * (torch.exp(-log_sigma_squared) * mse + log_sigma_squared)
    if args.reward_function == "mse":
        loss = torch.sum(mse)
    elif args.reward_function == "ama":
        loss = torch.sum(loss)
    if gradient_update % 1000 == 0:
        print(f"gradient_update {gradient_update} loss {loss}")
    loss.backward()
    optimizer.step()
    if args.reward_function == "mse":
        intrinsic_reward = torch.sum(mse)
    elif args.reward_function == "ama":
        intrinsic_reward = torch.sum(mse - torch.exp(log_sigma_squared))
    return network, optimizer, intrinsic_reward, torch.exp(log_sigma_squared)


def visualise_optimisation(network, x, y, gradient_update, reward_function):
    test_x = np.arange(-3, 6, 0.01)
    y_to_plot = []
    uncertainty_to_plot = []
    sigmas = []
    mses = []

    for an_x in test_x:
        mu, log_sigma_squared = network(torch.FloatTensor([an_x]).to(CUDA_DEVICE))
        y_to_plot.append(mu)
        sigmas.append((torch.exp(log_sigma_squared)))
        mses.append((mu - generating_function(an_x)) ** 2)

    y_to_plot = torch.FloatTensor(y_to_plot).cpu().numpy()
    plt.rc("font", family="serif")
    plt.rc("xtick", labelsize="medium")
    plt.rc("ytick")
    plt.title("AMA Reward Intuition")
    plt.scatter(
        x.cpu().numpy(),
        y.cpu().numpy(),
        c="Red",
        marker=".",
        s=1.0,
        label="Training Data",
    )
    plt.plot(test_x, y_to_plot, color="purple", label="Fitted Function")
    plt.plot(
        test_x,
        [generating_function(i) for i in test_x],
        color="red",
        label="True Underlying Function",
        alpha=0.2,
    )
    if args.reward_function == "ama":
        plt.fill_between(
            test_x,
            y_to_plot - (torch.FloatTensor(sigmas).detach().cpu().numpy()),
            y_to_plot + (torch.FloatTensor(sigmas).detach().cpu().numpy()),
            alpha=0.2,
            color="purple",
            label="Aleatoric Uncertainty",
        )
        plt.fill_between(
            test_x,
            y_to_plot - 2 * (torch.FloatTensor(sigmas).detach().cpu().numpy()),
            y_to_plot + 2 * (torch.FloatTensor(sigmas).detach().cpu().numpy()),
            alpha=0.2,
            color="purple",
        )
        plt.fill_between(
            test_x,
            y_to_plot
            - np.sqrt(
                np.clip(
                    (
                        torch.FloatTensor(mses).detach().cpu().numpy()
                        - torch.FloatTensor(sigmas).detach().cpu().numpy()
                    ),
                    a_min=0,
                    a_max=None,
                )
            ),
            y_to_plot
            + np.sqrt(
                np.clip(
                    (
                        torch.FloatTensor(mses).detach().cpu().numpy()
                        - torch.FloatTensor(sigmas).detach().cpu().numpy()
                    ),
                    a_min=0,
                    a_max=None,
                )
            ),
            alpha=0.2,
            color="orange",
            label="AMA Reward",
        )
        plt.fill_between(
            test_x,
            y_to_plot
            - 2
            * np.sqrt(
                np.clip(
                    (
                        torch.FloatTensor(mses).detach().cpu().numpy()
                        - torch.FloatTensor(sigmas).detach().cpu().numpy()
                    ),
                    a_min=0,
                    a_max=None,
                )
            ),
            y_to_plot
            + 2
            * np.sqrt(
                np.clip(
                    (
                        torch.FloatTensor(mses).detach().cpu().numpy()
                        - torch.FloatTensor(sigmas).detach().cpu().numpy()
                    ),
                    a_min=0,
                    a_max=None,
                )
            ),
            alpha=0.2,
            color="orange",
        )
    elif args.reward_function == "mse":
        plt.fill_between(
            test_x,
            y_to_plot
            - np.sqrt(
                np.clip(
                    (torch.FloatTensor(mses).detach().cpu().numpy()),
                    a_min=0,
                    a_max=None,
                )
            ),
            y_to_plot
            + np.sqrt(
                np.clip(
                    (torch.FloatTensor(mses).detach().cpu().numpy()),
                    a_min=0,
                    a_max=None,
                )
            ),
            alpha=0.2,
            color="orange",
            label="MSE Reward",
        )
        plt.fill_between(
            test_x,
            y_to_plot
            - 2
            * np.sqrt(
                np.clip(
                    (torch.FloatTensor(mses).detach().cpu().numpy()),
                    a_min=0,
                    a_max=None,
                )
            ),
            y_to_plot
            + 2
            * np.sqrt(
                np.clip(
                    (torch.FloatTensor(mses).detach().cpu().numpy()),
                    a_min=0,
                    a_max=None,
                )
            ),
            alpha=0.2,
            color="orange",
        )
    plt.ylim(-6, 6)
    plt.legend("upper left")
    plt.savefig(f"data/result_{gradient_update:0>8}.png")
    plt.close()


def compile_into_gif(repeat):
    import subprocess

    process = subprocess.Popen(
        f"convert -delay 10 -loop 0 data/*.png animation_{repeat}.gif",
        shell=True,
        stdout=subprocess.PIPE,
    )
    process.wait()


def main(args):
    agent = Agent(action_space=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    action_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: 0,
    }
    env = BanditEnvBase(obs_size=32)
    for repeat in range(1):
        network, optimizer = build_model()
        intrinsic_reward = torch.zeros(0)
        for step in range(args.update_steps):
            action = agent.step(intrinsic_reward.detach().cpu().numpy())
            action_counts[action] += 1
            (x, y), _ = env.step(action)
            x = torch.FloatTensor(x).to(CUDA_DEVICE)
            y = torch.FloatTensor(y).to(CUDA_DEVICE)
            network, optimizer, intrinsic_reward, uncertainty = update_model(
                network, optimizer, x, y, step, args.reward_function
            )
            wandb.log(
                {
                    "overall uncertainty": torch.clamp(
                        torch.mean(uncertainty.detach().cpu()), -3, 3
                    )
                }
            )
            if action in noisy_actions:
                wandb.log(
                    {
                        "noisy uncertainty": torch.clamp(
                            torch.mean(uncertainty.detach().cpu()), -3, 3
                        )
                    }
                )
            elif action in stable_actions:
                wandb.log(
                    {
                        "stable uncertainty": torch.clamp(
                            torch.mean(uncertainty.detach().cpu()), -3, 3
                        )
                    }
                )
            # if step % args.plot_frequency == 0:
            # #    visualise_optimisation(network, x, y, step, args.reward_function)
            print("Action counts", action_counts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_function", type=str)
    parser.add_argument("--update_steps", type=int)
    parser.add_argument("--plot_frequency", type=int)
    args = parser.parse_args()
    wandb.init(project="bandit")
    wandb.config.update(args)
    main(args)
