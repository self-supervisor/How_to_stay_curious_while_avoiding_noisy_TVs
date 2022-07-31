#!/usr/bin/env python
import numpy as np
import wandb
from gym import spaces
import os
import random
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
try:
    from OpenGL import GLU
except:
    print("no OpenGL.GLU")
import functools
import os.path as osp
from functools import partial

import gym
import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
from mpi4py import MPI

from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dynamics import Dynamics, UNet
from utils import random_agent_ob_mean_std
from wrappers import (
    MontezumaInfoWrapper,
    make_mario_env,
    make_robo_pong,
    make_robo_hockey,
    make_multi_pong,
    AddRandomStateToInfo,
    MaxAndSkipEnv,
    ProcessFrame84,
    ExtraTimeLimit,
    StickyActionEnv,
    NoisyTVEnvWrapper,
    NoisyTVEnvWrapperCIFAR,
    NoisyTVEnvWrapperMario,
    NoisyTVEnvWrapperMarioCIFAR,
    ActionLoggingWrapper,
)
from cifar10_web import cifar10
from load_cifar_10 import load_cifar_10_data
# from https://github.com/snatch59/load-cifar-10.git
cifar_train_data, _, _, _, _, _, _ = load_cifar_10_data('cifar-10-batches-py')

def get_random_cifar():
    image = cifar_train_data[random.randrange(0, len(cifar_train_data))]
    return image

def start_experiment(**args):
    make_env = partial(make_env_all_params, add_monitor=True, args=args)

    trainer = Trainer(
        make_env=make_env,
        num_timesteps=args["num_timesteps"],
        hps=args,
        envs_per_process=args["envs_per_process"],
    )
    log, tf_sess = get_experiment_environment(**args)
    with log, tf_sess:
        writer = tf.summary.FileWriter("./graphs", graph=tf_sess.graph)
        logdir = logger.get_dir()
        print("results will be saved to ", logdir)
        trainer.train()


class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()

        self.policy = CnnPolicy(
            scope="pol",
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            hidsize=512,
            feat_dim=512,
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nl=tf.nn.leaky_relu,
        )

        self.feature_extractor = {
            "none": FeatureExtractor,
            "idf": InverseDynamics,
            "vaesph": partial(VAE, spherical_obs=True),
            "vaenonsph": partial(VAE, spherical_obs=False),
            "pix2pix": JustPixels,
        }[hps["feat_learning"]]
        self.feature_extractor = self.feature_extractor(
            policy=self.policy,
            features_shared_with_policy=False,
            feat_dim=512,
            layernormalize=hps["layernorm"],
        )

        self.dynamics = Dynamics if hps["feat_learning"] != "pix2pix" else UNet
        self.dynamics = self.dynamics(
            auxiliary_task=self.feature_extractor,
            predict_from_pixels=hps["dyn_from_pixels"],
            feat_dim=512,
            ama=hps["ama"],
            uncertainty_penalty=hps["uncertainty_penalty"],
            clip_ama=hps["clip_ama"],
            clip_val=hps["clip_val"],
            reward_scaling=hps["reward_scaling"],
            abs_ama=hps["abs_ama"],
            eta=hps["eta"],
        )
        self.agent = PpoOptimizer(
            scope="ppo",
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps["use_news"],
            gamma=hps["gamma"],
            lam=hps["lambda"],
            nepochs=hps["nepochs"],
            nminibatches=hps["nminibatches"],
            lr=hps["lr"],
            cliprange=0.1,
            nsteps_per_seg=hps["nsteps_per_seg"],
            nsegs_per_env=hps["nsegs_per_env"],
            ent_coef=hps["ent_coeff"],
            normrew=hps["norm_rew"],
            normadv=hps["norm_adv"],
            ext_coeff=hps["ext_coeff"],
            int_coeff=hps["int_coeff"],
            dynamics=self.dynamics,
            args=hps,
        )

        self.agent.to_report["aux"] = tf.reduce_mean(self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report["aux"]
        self.agent.to_report["dyn_loss"] = tf.reduce_mean(self.dynamics.loss[0])
        self.agent.total_loss += self.agent.to_report["dyn_loss"]
        self.agent.to_report["feat_var"] = tf.reduce_mean(
            tf.nn.moments(self.feature_extractor.features, [0, 1])[1]
        )

    def _set_env_vars(self):
        env = self.make_env(0, add_monitor=False)
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        self.envs = [
            functools.partial(self.make_env, i) for i in range(self.envs_per_process)
        ]

    def train(self):
        import random

        self.agent.start_interaction(
            self.envs, nlump=self.hps["nlumps"], dynamics=self.dynamics
        )
        count = 0
        while True:
            count += 1
            info = self.agent.step()
            if info["update"]:
                logger.logkvs(info["update"])
                logger.dumpkvs()
            if self.hps["feat_learning"] == "pix2pix":
                making_video = random.choice(99 * [False] + [True])
            else:
                making_video = False
            self.agent.rollout.making_video = making_video
            for a_key in info.keys():
                wandb.log(info[a_key])
            from numpy import inf, nan

            # be careful when examining logs because of this inf problem
            mean_sigma = np.mean(self.agent.rollout.buf_sigmas)
            if mean_sigma == inf or mean_sigma == -inf or mean_sigma == nan:
                mean_sigma = 0
            wandb.log({"average_sigma": mean_sigma})
            if self.agent.rollout.stats["tcount"] > self.num_timesteps:
                break

        self.agent.stop_interaction()


def make_env_all_params(rank, add_monitor, args):
    if args["env_kind"] == "atari":
        assert args["env"] != "mario"
        env = gym.make(args["env"])
        assert "NoFrameskip" in env.spec.id
        # from self-supervised exploration via disagreement
        if args["stickyAtari"] == "true":
            env = StickyActionEnv(env)
        env._max_episode_steps = args["max_episode_steps"] * 4
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        env = ExtraTimeLimit(env, args["max_episode_steps"])
        if "Montezuma" in args["env"]:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
        if args["noisy_tv"] == "true":
            if args["cifar"] == "true":
                env = NoisyTVEnvWrapperCIFAR(env, get_random_cifar)
            else:
                env = NoisyTVEnvWrapper(env)
        # assert env.action_space == spaces.Discrete(7)
    elif args["env_kind"] == "mario":
        assert args["env"] == "mario"
        env = make_mario_env()
        if args["noisy_tv"] == "true":
            if args["cifar"] == "true":
                env = NoisyTVEnvWrapperMarioCIFAR(env, get_random_cifar)
            else:
                env = NoisyTVEnvWrapperMario(env)
    elif args["env_kind"] == "retro_multi":
        env = make_multi_pong()
    elif args["env_kind"] == "robopong":
        if args["env"] == "pong":
            env = make_robo_pong()
        elif args["env"] == "hockey":
            env = make_robo_hockey()

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), "%.2i" % rank))
    return env


def get_experiment_environment(**args):
    from utils import setup_mpi_gpus, setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed

    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    setup_mpi_gpus()

    logger_context = logger.scoped_configure(
        dir=None,
        format_strs=["stdout", "log", "csv"]
        if MPI.COMM_WORLD.Get_rank() == 0
        else ["log"],
    )
    tf_context = setup_tensorflow_session()
    return logger_context, tf_context


def add_environments_params(parser):
    parser.add_argument(
        "--env", help="environment ID", default="BankHeistNoFrameskip-v4", type=str
    )
    parser.add_argument(
        "--max-episode-steps",
        help="maximum number of timesteps for episode",
        default=4500,
        type=int,
    )
    parser.add_argument("--env_kind", type=str, default="atari")
    parser.add_argument("--stickyAtari", type=str, default="false")
    parser.add_argument("--noisy_tv", type=str, default="false")
    parser.add_argument("--cifar", type=str, default="false")
    parser.add_argument("--noop_max", type=int, default=30)


def add_optimization_params(parser):
    parser.add_argument("--lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--nminibatches", type=int, default=8)
    parser.add_argument("--norm_adv", type=int, default=1)
    parser.add_argument("--norm_rew", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ent_coeff", type=float, default=0.001)
    parser.add_argument("--nepochs", type=int, default=3)
    parser.add_argument("--num_timesteps", type=int, default=int(1e7))


def add_rollout_params(parser):
    parser.add_argument("--nsteps_per_seg", type=int, default=128)
    parser.add_argument("--nsegs_per_env", type=int, default=1)
    parser.add_argument("--envs_per_process", type=int, default=128)
    parser.add_argument("--nlumps", type=int, default=1)


if __name__ == "__main__":
    wandb.init(project="burda_pathak_fork")
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_environments_params(parser)
    add_optimization_params(parser)
    add_rollout_params(parser)

    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--seed", help="RNG seed", type=int, default=0)
    parser.add_argument("--dyn_from_pixels", type=int, default=0)
    parser.add_argument("--use_news", type=int, default=0)
    parser.add_argument("--ext_coeff", type=float, default=0.0)
    parser.add_argument("--int_coeff", type=float, default=1.0)
    parser.add_argument("--layernorm", type=int, default=0)
    parser.add_argument("--uncertainty_penalty", type=float, default=1)
    parser.add_argument("--clip_ama", type=str, default="false")
    parser.add_argument("--ama", type=str, default="true")
    parser.add_argument("--abs_ama", type=str, default="false")
    parser.add_argument("--clip_val", type=float, default=1e6)
    parser.add_argument("--reward_scaling", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument(
        "--feat_learning",
        type=str,
        default="none",
        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"],
    )

    args = parser.parse_args()
    wandb.config.update(args)
    start_experiment(**args.__dict__)

