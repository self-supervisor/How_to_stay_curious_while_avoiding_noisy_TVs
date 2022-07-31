import os, logging, time
import gym
from gym_recording.wrappers import TraceRecordingWrapper
from gym_recording.playback import scan_recorded_traces
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def test_trace_recording():
    counts = [0, 0]
    def handle_ep(observations, actions, rewards):
        print("I am alive")
        counts[0] += 1
        counts[1] += observations.shape[0]
        logger.debug('Observations.shape={}, actions.shape={}, rewards.shape={}', observations.shape, actions.shape, rewards.shape)

    scan_recorded_traces("/tmp/openai-2021-03-02-14-55-06-981879/", handle_ep)
