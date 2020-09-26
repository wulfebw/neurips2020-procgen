import copy

from gym.wrappers import TransformReward
import numpy as np
from ray.rllib.env.atari_wrappers import FrameStack
from ray.tune import registry

from envs.frame_diff import FrameDiff
from envs.frame_stack_phase_correlation import FrameStackPhaseCorrelation
from envs.grayscale import Grayscale
from envs.procgen_env_wrapper import ProcgenEnvWrapper


def wrap_procgen(env,
                 frame_diff=False,
                 frame_diff_options={},
                 frame_stack=False,
                 frame_stack_options={},
                 frame_stack_phase_correlation=False,
                 frame_stack_phase_correlation_options={},
                 normalize_reward=False,
                 grayscale=False):
    env_name = env.env_name
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
    if normalize_reward:
        raise NotImplementedError("Use built in min/max returns")
        env_max_return = PROCGEN_MAX_RETURN[env_name]
        env_reward_scale = 10.0 / env_max_return
        env = TransformReward(env, lambda r: r * env_reward_scale)
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
