import numpy as np
import argparse
import time
import numpy
import torch
import utils
from .conversion_utils import (
    scale_for_autoencoder,
    convert_representation_to_rgb,
    convert_obs_to_rgb,
)

# from gym_minigrid.wrappers import *

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env", required=True, help="name of the environment to be run (REQUIRED)"
)
parser.add_argument(
    "--model", required=True, help="name of the trained model (REQUIRED)"
)
parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
parser.add_argument(
    "--shift",
    type=int,
    default=0,
    help="number of times the environment is reset at the beginning (default: 0)",
)
parser.add_argument(
    "--argmax",
    action="store_true",
    default=False,
    help="select the action with highest probability (default: False)",
)
parser.add_argument(
    "--pause",
    type=float,
    default=0.1,
    help="pause duration between two consequent actions of the agent (default: 0.1)",
)
parser.add_argument(
    "--gif", type=str, default=None, help="store output as gif with the given filename"
)
parser.add_argument(
    "--episodes", type=int, default=1000000, help="number of episodes to visualize"
)
parser.add_argument(
    "--memory", action="store_true", default=False, help="add a LSTM to the model"
)
parser.add_argument(
    "--text", action="store_true", default=False, help="add a GRU to the model"
)
parser.add_argument(
    "--autoencoder_path", type=str, help="path to trained autoencoder model"
)
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
autoencoder = torch.load(args.autoencoder_path)
agent = utils.Agent(
    env.observation_space,
    env.action_space,
    model_dir,
    autoencoder=autoencoder,
    device=device,
    argmax=args.argmax,
    use_memory=args.memory,
    use_text=args.text,
)
print("Agent loaded\n")

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []
    obs_array = []
    predicted_obs = []
    predicted_uncertainty = []
    combined = []

# Create a window to view the environment
env.render("human")
for episode in range(args.episodes):
    obs = env.reset()
    action = 0

    while True:

        env.render("human")
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        a_predicted_obs, a_predicted_unc = agent.autoencoder(
            scale_for_autoencoder(
                torch.FloatTensor(obs["image"]).unsqueeze(0).permute(0, 3, 1, 2),
                normalise=True,
            ),
            torch.tensor(action, dtype=torch.int).unsqueeze(0),
        )
        predicted_obs.append(
            a_predicted_obs.permute(0, 3, 2, 1).squeeze(0).detach().numpy()
        )
        predicted_uncertainty.append(
            a_predicted_unc.permute(0, 3, 2, 1).squeeze(0).detach().numpy()
        )
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        obs_array.append(
            (
                scale_for_autoencoder(
                    (torch.FloatTensor(obs["image"]).unsqueeze(0).permute(0, 3, 1, 2)),
                    normalise=True,
                )
                .squeeze(0)
                .detach()
                .numpy()
            )
        )
        # combined.append(
        #    np.hstack(
        #        (
        #            (convert_obs_to_rgb(obs)),
        #            convert_representation_to_rgb(
        #                a_predicted_obs.squeeze(0).detach().numpy()
        #            ),
        #            convert_representation_to_rgb(
        #                a_predicted_unc.squeeze(0).detach().numpy()
        #            ),
        #        )
        #    )
        # )
        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            break

    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    # write_gif(numpy.array(frames), args.gif + ".gif", fps=1 / args.pause)
    # write_gif(numpy.array(obs_array), args.gif + "_obs_array.gif", fps=1 / args.pause)
    # write_gif(
    #    numpy.array(predicted_obs), args.gif + "_predicted_obs.gif", fps=1 / args.pause
    # )
    # write_gif(
    #    numpy.array(predicted_uncertainty),
    #    args.gif + "_predicted_uncertainty.gif",
    #    fps=1 / args.pause,
    # )
    # write_gif(
    #    np.array(combined), args.gif + "_combined.gif", fps=1 / args.pause,
    # )
    np.save("predicted_obs.npy", predicted_obs)
    np.save("predicted_uncertainty.npy", predicted_uncertainty)
    np.save("obs.npy", obs_array)
    np.save("frames.npy", frames)
    print("Done.")
