import copy

from gym.wrappers import TransformReward
import numpy as np
from ray.rllib.env.atari_wrappers import FrameStack
from ray.tune import registry

from envs.frame_diff import FrameDiff
from envs.frame_stack_phase_correlation import FrameStackPhaseCorrelation
from envs.grayscale import Grayscale
from envs.mixed_grayscale_color_frame_stack import MixedGrayscaleColorFrameStack
from envs.procgen_env_wrapper import ProcgenEnvWrapper
from envs.reward_normalization_wrapper import RewardNormalizationWrapper
from envs.state_occupancy_counter import StateOccupancyCounter


def wrap_procgen(env,
                 frame_diff=False,
                 frame_diff_options={},
                 frame_stack=False,
                 frame_stack_options={},
                 frame_stack_phase_correlation=False,
                 frame_stack_phase_correlation_options={},
                 normalize_reward=False,
                 normalize_reward_options={},
                 grayscale=False,
                 mixed_grayscale_color=False,
                 mixed_grayscale_color_options={},
                 count_state_occupancy=False):
    env_name = env.env_name
    if count_state_occupancy:
        env = StateOccupancyCounter(env)
    if frame_diff:
        env = FrameDiff(env, **frame_diff_options)
    if grayscale:
        assert not frame_diff
        assert not frame_stack_phase_correlation
        env = Grayscale(env)
    if frame_stack:
        env = FrameStack(env, **frame_stack_options)
    if frame_stack_phase_correlation:
        env = FrameStackPhaseCorrelation(env, **frame_stack_phase_correlation_options)
    if mixed_grayscale_color:
        assert not frame_diff
        assert not grayscale
        assert not frame_stack_phase_correlation
        assert not frame_stack
        env = MixedGrayscaleColorFrameStack(env, **mixed_grayscale_color_options)
    if normalize_reward:
        env = RewardNormalizationWrapper(env, **normalize_reward_options)
    return env


def create_env(config):
    config = copy.deepcopy(config)
    if "env_wrapper_options" in config:
        env_wrapper_options = config["env_wrapper_options"]
        del config["env_wrapper_options"]
    else:
        env_wrapper_options = {}
    env = ProcgenEnvWrapper(config)
    env = wrap_procgen(env, **env_wrapper_options)
    return env


registry.register_env("custom_procgen_env_wrapper", create_env)
