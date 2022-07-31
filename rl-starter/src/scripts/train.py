import utils
import time


def log_to_wandb(logs, start_time, update_start_time, update_end_time):
    fps = logs["num_frames"] / (update_end_time - update_start_time)
    wandb.log({"fps": fps})
    duration = int(time.time() - start_time)
    wandb.log({"duration": duration})
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    wandb.log({"return_per_episode": return_per_episode})
    rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
    wandb.log({"rreturn_per_episode": rreturn_per_episode})
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
    wandb.log({"number_frames_per_episode": num_frames_per_episode})
    for a_key in rreturn_per_episode.keys():
        wandb.log({"rreturn_" + a_key: rreturn_per_episode[a_key]})
        wandb.log({"num_frames_" + a_key: num_frames_per_episode[a_key]})
    wandb.log({"intrinsic_rewards": logs["intrinsic_rewards"].mean().item()})
    wandb.log({"uncertainties": logs["uncertainties"].mean().item()})
    wandb.log({"novel_states_visited": logs["novel_states_visited"].max().item()})
    wandb.log({"entropy": logs["entropy"]})
    wandb.log({"value": logs["value"]})
    wandb.log({"policy_loss": logs["policy_loss"]})
    wandb.log({"value_loss": logs["value_loss"]})
    wandb.log({"grad_norm": logs["grad_norm"]})


def tuner(icm_lr, reward_weighting, normalise_rewards, args):
    import argparse
    import datetime
    import torch
    import torch_ac
    import tensorboardX
    import sys
    import numpy as np
    from model import ACModel
    from .a2c import A2CAlgo

    # from .ppo import PPOAlgo

    frames_to_visualise = 200
    # Parse arguments

    args.mem = args.recurrence > 1

    def make_exploration_heatmap(args, plot_title):
        import numpy as np
        import matplotlib.pyplot as plt

        visitation_counts = np.load(
            f"{args.model}_visitation_counts.npy", allow_pickle=True
        )
        plot_title = str(np.count_nonzero(visitation_counts)) + args.model
        plt.imshow(np.log(visitation_counts))
        plt.colorbar()
        plt.title(plot_title)
        plt.savefig(f"{plot_title}_visitation_counts.png")

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"
    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")
    # Load environments

    envs = []

    for i in range(16):
        an_env = utils.make_env(
            args.env, int(args.frames_before_reset), int(args.environment_seed)
        )
        envs.append(an_env)
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model

    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    # adapted from impact driven RL
    from .models import AutoencoderWithUncertainty

    autoencoder = AutoencoderWithUncertainty(observation_shape=(7, 7, 3)).to(device)

    autoencoder_opt = torch.optim.Adam(
        autoencoder.parameters(), lr=icm_lr, weight_decay=0
    )
    if args.algo == "a2c":
        algo = A2CAlgo(
            envs,
            acmodel,
            autoencoder,
            autoencoder_opt,
            args.uncertainty,
            args.noisy_tv,
            args.curiosity,
            args.randomise_env,
            args.uncertainty_budget,
            args.environment_seed,
            reward_weighting,
            normalise_rewards,
            args.frames_before_reset,
            device,
            args.frames_per_proc,
            args.discount,
            args.lr,
            args.gae_lambda,
            args.entropy_coef,
            args.value_loss_coef,
            args.max_grad_norm,
            args.recurrence,
            args.optim_alpha,
            args.optim_eps,
            preprocess_obss,
            None,
            args.random_action,
        )
    elif args.algo == "ppo":
        algo = PPOAlgo(
            envs,
            acmodel,
            autoencoder,
            autoencoder_opt,
            args.uncertainty,
            args.noisy_tv,
            args.curiosity,
            args.randomise_env,
            args.uncertainty_budget,
            args.environment_seed,
            reward_weighting,
            normalise_rewards,
            device,
            args.frames_per_proc,
            args.discount,
            args.lr,
            args.gae_lambda,
            args.entropy_coef,
            args.value_loss_coef,
            args.max_grad_norm,
            args.recurrence,
            args.optim_eps,
            args.clip_eps,
            args.epochs,
            args.batch_size,
            preprocess_obss,
        )

    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        log_to_wandb(logs, start_time, update_start_time, update_end_time)

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += [
                "intrinsic_rewards",
                "uncertainties",
                "novel_states_visited",
                "entropy",
                "value",
                "policy_loss",
                "value_loss",
                "grad_norm",
            ]
            data += [
                logs["intrinsic_rewards"].mean().item(),
                logs["uncertainties"].mean().item(),
                logs["novel_states_visited"].mean().item(),
                logs["entropy"],
                logs["value"],
                logs["policy_loss"],
                logs["value_loss"],
                logs["grad_norm"],
            ]
            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f}".format(
                    *data
                )
            )
        # Save status
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {
                "num_frames": num_frames,
                "update": update,
                "model_state": acmodel.state_dict(),
                "optimizer_state": algo.optimizer.state_dict(),
            }
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
    return


if __name__ == "__main__":
    import wandb
    import argparse

    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument(
        "--algo", required=True, help="algorithm to use: a2c | ppo (REQUIRED)"
    )
    parser.add_argument("--environment_seed", help="Seed for environment reset")
    parser.add_argument(
        "--uncertainty_budget", help="how much uncertainty to allow", default=1
    )
    parser.add_argument("--randomise_env", help="whether to use curiosity")
    parser.add_argument("--curiosity", help="whether to use curiosity")
    parser.add_argument(
        "--env", required=True, help="name of the environment to train on (REQUIRED)"
    )
    parser.add_argument(
        "--model", default=None, help="name of the model (default: {ENV}_{ALGO}_{TIME})"
    )
    # parser.add_argument("--normalising_rewards")
    parser.add_argument("--seed", type=int, default=2, help="random seed (default: 1)")
    parser.add_argument("--visualizing")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="number of updates between two logs (default: 1)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="number of updates between two saves (default: 10, 0 means no saving)",
    )
    parser.add_argument(
        "--procs", type=int, default=16, help="number of processes (default: 128)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10 ** 7,
        help="number of frames of training (default: 1e7)",
    )

    ## Parameters for main algorithm
    parser.add_argument(
        "--epochs", type=int, default=4, help="number of epochs for PPO (default: 4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="batch size for PPO (default: 256)"
    )
    parser.add_argument(
        "--frames-per-proc",
        type=int,
        default=None,
        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)",
    )
    parser.add_argument(
        "--discount", type=float, default=0.99, help="discount factor (default: 0.99)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss term coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="maximum norm of gradient (default: 0.5)",
    )
    parser.add_argument(
        "--optim-eps",
        type=float,
        default=1e-8,
        help="Adam and RMSprop optimizer epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--optim-alpha",
        type=float,
        default=0.99,
        help="RMSprop optimizer alpha (default: 0.99)",
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.2,
        help="clipping epsilon for PPO (default: 0.2)",
    )
    parser.add_argument(
        "--recurrence",
        type=int,
        default=1,
        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        default=False,
        help="add a GRU to the model to handle text input",
    )
    parser.add_argument(
        "--uncertainty", help="whether to shape rewards with aleatoric uncertainties"
    )
    parser.add_argument(
        "--normalise_rewards",
        help="whether to normalise rewards with moving average of mean and variance",
    )
    parser.add_argument("--icm_lr", help="icm learning rate")
    parser.add_argument("--reward_weighting", help="factor to scale rewards by")
    parser.add_argument("--noisy_tv", help="whether to add a noisy tv or not")
    parser.add_argument(
        "--random_action",
        help="naive policy of simply selecting random actions from action space.",
    )
    parser.add_argument("--frames_before_reset")
    args = parser.parse_args()

    wandb.init(project="minigrid")
    wandb.config.update(args)
    novel_states = tuner(
        float(args.icm_lr), float(args.reward_weighting), args.normalise_rewards, args
    )
    import csv

    with open(
        str(args.model).split("_seed")[0] + "_" + str(args.seed) + ".csv", "a"
    ) as fp:
        wr = csv.writer(fp)
        wr.writerow([float(args.icm_lr), (args.reward_weighting), novel_states])
