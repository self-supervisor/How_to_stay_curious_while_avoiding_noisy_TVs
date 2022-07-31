import wandb
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

CUDA_DEVICE = "cpu"
# CUDA_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stable_reward_actions = [2, 3, 4, 5]
noisy_reward_actions = [8, 9, 10, 11]


def build_ensemble(size):
    networks = []
    optimizers = []
    for _ in range(size):
        a_network, an_optimizer = build_model()
        networks.append(a_network)
        optimizers.append(an_optimizer)
    return networks, optimizers


def build_model():
    network = Net()
    network.to(CUDA_DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=0.0001)
    network.train()
    return network, optimizer


def update_model(network, optimizer, x, y, epoch):
    optimizer.zero_grad()
    mu = network(x.reshape(32, 1))
    mse = F.mse_loss(mu, y.reshape(32, 1), reduction="none")
    loss = torch.sum(mse)
    print(f"epoch {epoch} loss {loss}")
    loss.backward()
    optimizer.step()
    return network, optimizer, mu


def compute_ensemble_reward(mu_list):
    mu_list = torch.stack(mu_list)
    intrinsic_reward = torch.mean(torch.var(mu_list, 0))
    return intrinsic_reward


def visualise_optimisation(networks, x, y, epoch):
    test_x = np.arange(-3, 6, 0.01)
    y_to_plot = []
    uncertainty_to_plot = []
    sigmas = []
    mses = []

    mu_list_of_lists = []

    for an_x in test_x:
        mu_list = []
        for i, network in enumerate(networks):
            networks[i] = networks[i].eval()
            mu = networks[i](torch.FloatTensor([an_x]).to(CUDA_DEVICE))
            mu_list.append(mu)
            # mses.append((mu - generating_function(an_x)) ** 2)
        mu_list_of_lists.append(mu_list)

    epistemic_uncertainty = [
        compute_ensemble_reward(ensemble_predictions)
        for ensemble_predictions in mu_list_of_lists
    ]
    y_to_plot = torch.FloatTensor(
        [torch.mean(torch.stack(i)) for i in mu_list_of_lists]
    )
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
    plt.fill_between(
        test_x,
        y_to_plot
        - 4 * (torch.FloatTensor(epistemic_uncertainty).detach().cpu().numpy()),
        y_to_plot
        + 4 * (torch.FloatTensor(epistemic_uncertainty).detach().cpu().numpy()),
        alpha=0.2,
        color="purple",
        label="Epistemic Uncertainty",
    )
    plt.ylim(-6, 6)
    plt.legend(loc="lower right")
    plt.savefig(f"data/result_{epoch:0>8}.png")
    plt.close()


def compile_into_gif(repeat, reward_function):
    import time
    import subprocess

    unix_timestamp = time.time()
    process = subprocess.Popen(
        f"convert -delay 10 -loop 0 data/*.png animation_{reward_function}_time_{unix_timestamp}.gif",
        shell=True,
        stdout=subprocess.PIPE,
    )
    process.wait()


def main(args):
    import wandb

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
    intrinsic_reward_list = []
    action_list = []
    for repeat in range(1):
        networks, optimizers = build_ensemble(size=3)
        intrinsic_reward = torch.zeros(0)
        for step in range(args.update_steps):
            action = agent.step(intrinsic_reward.detach().cpu().numpy())
            action_counts[action] += 1
            (x, y), _ = env.step(action)
            x = torch.FloatTensor(x).to(CUDA_DEVICE)
            y = torch.FloatTensor(y).to(CUDA_DEVICE)
            mu_list = []
            for i, network in enumerate(networks):
                networks[i] = networks[i].train()
                networks[i], optimizers[i], mu = update_model(
                    networks[i], optimizers[i], x, y, step
                )
                mu_list.append(mu)
            intrinsic_reward = compute_ensemble_reward(mu_list)
            intrinsic_reward_list.append(intrinsic_reward)
            action_list.append(action)
            wandb.log({"overall_intrinsic_reward": intrinsic_reward})
            if action in stable_reward_actions:
                wandb.log({"stable_intrinsic_reward": intrinsic_reward})
            elif action in noisy_reward_actions:
                wandb.log({"noisy_intrinsic_reward": intrinsic_reward})
            # if step % args.plot_frequency == 0:
            #    visualise_optimisation(networks, x, y, step)
            print("Action counts", action_counts)

        import datetime
        import time

        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        compile_into_gif(repeat, reward_function=args.reward_function)
        np.save(
            f"action_counts_reward_function_{timestamp}.npy",
            action_counts,
            allow_pickle=True,
        )
        np.save(
            f"action_list_{timestamp}.npy",
            action_list,
        )
        np.save(
            f"intrinsic_reward_list_{timestamp}.npy",
            intrinsic_reward_list,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_function", type=str)
    parser.add_argument("--update_steps", type=int)
    parser.add_argument("--plot_frequency", type=int)
    args = parser.parse_args()
    wandb.init(project="bandit")
    main(args)
