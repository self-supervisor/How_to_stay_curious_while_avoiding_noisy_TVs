import pytest
import matplotlib.pyplot as plt
import numpy as np
import gym
from .wrappers import (
    MontezumaInfoWrapper,
    AddRandomStateToInfo,
    MaxAndSkipEnv,
    ProcessFrame84,
    ExtraTimeLimit,
    StickyActionEnv,
    NoisyTVEnvWrapper,
)
from .baselines.common.atari_wrappers import FrameStack


@pytest.fixture(
    scope="module", params=["SpaceInvadersNoFrameskip-v4", "BreakoutNoFrameskip-v4"]
)
def atari_env(request):
    env = gym.make(request.param)
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrame84(env, crop=False)
    env = FrameStack(env, 4)
    env = AddRandomStateToInfo(env)
    return env


def test_extra_action_added(atari_env):
    from .wrappers import NoisyTVEnvWrapper

    actions_before_wrap = atari_env.env.action_space.n
    atari_env = NoisyTVEnvWrapper(atari_env)
    actions_after_wrap = atari_env.env.action_space.n
    assert actions_before_wrap + 1 == actions_after_wrap


def test_noisy_TV_turns_on(atari_env):
    from .wrappers import NoisyTVEnvWrapper
    from scipy.stats import norm
    import math

    _ = atari_env.reset()
    output = atari_env.step(atari_env.action_space.sample() - 1)
    obs = output[0]
    assert math.isclose(np.mean(obs), 125, rel_tol=0.05) == False
    plt.imshow(obs)
    plt.savefig("before_noisy_TV.png")
    plt.close()

    atari_env = NoisyTVEnvWrapper(atari_env)
    output = atari_env.step(atari_env.noisy_action)
    obs = output[0]
    assert math.isclose(np.mean(obs), 125, rel_tol=0.05) == True
    plt.imshow(obs.__array__()[:,:,0])
    plt.savefig("after_noisy_TV.png")
    plt.close()
