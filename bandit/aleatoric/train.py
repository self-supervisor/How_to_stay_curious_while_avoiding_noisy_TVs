from dataset import generating_function
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from model import Net
from dataset import get_data
import torch.nn.functional as F
from copy import deepcopy

CUDA_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_data_and_model():
    x, y = get_data()
    x = torch.FloatTensor(x).to(CUDA_DEVICE)
    y = torch.FloatTensor(y).to(CUDA_DEVICE)

    network = Net()
    network.to(CUDA_DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    network.train()
    return network, optimizer, (x, y)


def update_model(network, optimizer, x, y, epoch):
    optimizer.zero_grad()
    network_copy = deepcopy(network)
    mu, log_sigma_squared = network(x.reshape(1000, 1))
    mse = F.mse_loss(mu, y.reshape(1000, 1), reduction="none")
    loss = 0.5 * (torch.exp(-log_sigma_squared) * mse + 0.5 * log_sigma_squared)
    loss = mse
    loss = torch.sum(loss)
    print(f"epoch {epoch} loss {loss}")
    loss.backward()
    optimizer.step()
    return network, optimizer


def visualise_optimisation(network, x, y, epoch):
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
        alpha=0.1,
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
    plt.ylim(-6, 6)
    plt.legend()
    plt.savefig(f"data/result_{epoch:0>8}.png")
    plt.close()


def compile_into_gif(repeat):
    import subprocess

    process = subprocess.Popen(
        f"convert -delay 30 -loop 0 data/*.png animation_{repeat}.gif",
        shell=True,
        stdout=subprocess.PIPE,
    )
    process.wait()


def main():
    for repeat in range(1):
        network, optimizer, (x, y) = build_data_and_model()
        for epoch in range(20000):
            network, optimizer = update_model(network, optimizer, x, y, epoch)
            if epoch % 300 == 0:
                visualise_optimisation(network, x, y, epoch)
        compile_into_gif(repeat)


if __name__ == "__main__":
    main()
