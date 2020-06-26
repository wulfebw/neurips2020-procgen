import copy

import numpy as np
from ray.tune import registry

from envs.frame_diff import FrameDiff
from envs.mean_normalize_image_obs import MeanNormalizeImageObs
from envs.procgen_env_wrapper import ProcgenEnvWrapper

PROCGEN_OBS_MEANS = {
    "bigfish": np.array([75, 131, 169]),
    "bossfight": np.array([44, 45, 45]),
    "caveflyer": np.array([147, 158, 157]),
    "chaser": np.array([113, 134, 123]),
    "climber": np.array([107, 118, 98]),
    "coinrun": np.array([146, 148, 129]),
    "dodgeball": np.array([114, 109, 84]),
    "fruitbot": np.array([118, 113, 88]),
    "heist": np.array([163, 172, 163]),
    "jumper": np.array([91, 126, 91]),
    "leaper": np.array([122, 123, 106]),
    "maze": np.array([176, 136, 91]),
    "miner": np.array([160, 122, 83]),
    "ninja": np.array([145, 147, 136]),
    "plunder": np.array([74, 119, 158]),
    "starpilot": np.array([46, 48, 43]),
}


def get_obs_mean_for_env(env, dtype=np.uint8):
    assert env in PROCGEN_OBS_MEANS, f"Invalid env name: {env}"
    return PROCGEN_OBS_MEANS[env].astype(dtype)


def get_diff_mean(grayscale, diff_mean_value=255 // 2, dtype=np.uint8):
    num_diff_channels = 1 if grayscale else 3
    return np.ones(num_diff_channels, dtype=dtype) * diff_mean_value


def get_obs_mean(env_name, frame_diff, frame_diff_options):
    obs_mean = get_obs_mean_for_env(env_name)
    if not frame_diff:
        return obs_mean

    diff_mean = get_diff_mean(frame_diff_options.get("grayscale", True))
    mean = np.concatenate((diff_mean, obs_mean))
    return mean


def wrap_procgen(env, frame_diff=True, frame_diff_options={}, normalize_obs=True):
    env_name = env.env_name
    if frame_diff:
        env = FrameDiff(env, **frame_diff_options)
    if normalize_obs:
        mean = get_obs_mean(env_name, frame_diff, frame_diff_options)
        env = MeanNormalizeImageObs(env, mean)
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
