#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym
import gym_minigrid
import pytest
import numpy as np


@pytest.fixture(scope="module",params=[1,2,3,100,89,2000])
def key_corridor_altered(request):
    from utils import make_env

    seed = request.param
    env = make_env("MiniGrid-KeyCorridorS6R3-v0", 8, seed)
    return env


def test_not_procedural(key_corridor_altered):
    import random

    initial_position = key_corridor_altered.agent_pos
    for _ in range(100):
        for _ in range(8):
            obs, reward, done, info = key_corridor_altered.step(random.randint(0,6))
            if done:
                key_corridor_altered.reset()
        assert np.array_equal(key_corridor_altered.agent_pos, initial_position)


