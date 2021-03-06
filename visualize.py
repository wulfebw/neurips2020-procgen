#!/usr/bin/env python

import argparse
import collections
import copy
import json
import os
from pathlib import Path
import pickle
import shelve

import cv2
import gym
import gym.wrappers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import ray
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
try:
    from ray.rllib.evaluation.episode import _flatten_action
except Exception:
    # For newer ray versions
    from ray.rllib.utils.space_utils import flatten_to_single_ndarray as _flatten_action

from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.deprecation import deprecation_warning
from ray.tune.utils import merge_dicts
from ray.tune.registry import get_trainable_cls

from utils.loader import load_envs, load_models, load_algorithms
"""
Note : This script has been adapted from :
    https://github.com/ray-project/ray/blob/master/rllib/rollout.py
"""

EXAMPLE_USAGE = """
Example Usage:

python ./rollout.py \
    /tmp/ray/checkpoint_dir/checkpoint-0 \
    --env procgen_env_wrapper \
    --run PPO \
    --episodes 100 
"""

# Register all necessary assets in tune registries
load_envs(os.getcwd())  # Load envs
load_models(os.getcwd())  # Load models
# Load custom algorithms
from algorithms import CUSTOM_ALGORITHMS
load_algorithms(CUSTOM_ALGORITHMS)


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(formatter_class=argparse.RawDescriptionHelpFormatter,
                            description="Roll out a reinforcement learning agent "
                            "given a checkpoint.",
                            epilog=EXAMPLE_USAGE)

    parser.add_argument("checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument("--run",
                                type=str,
                                required=True,
                                help="The algorithm or model to train. This may refer to the name "
                                "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
                                "user-defined trainable function or class registered in the "
                                "tune registry.")
    required_named.add_argument("--env", type=str, help="The gym environment to use.")
    parser.add_argument("--video-dir",
                        type=str,
                        default=None,
                        help="Specifies the directory into which videos of all episode "
                        "rollouts will be stored.")
    parser.add_argument("--steps",
                        default=10000,
                        help="Number of timesteps to roll out (overwritten by --episodes).")
    parser.add_argument("--episodes",
                        type=int,
                        default=0,
                        help="Number of complete episodes to roll out (overrides --steps).")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument("--config",
                        default="{}",
                        type=json.loads,
                        help="Algorithm-specific configuration (e.g. env, hyperparams). "
                        "Gets merged with loaded configuration from checkpoint file and "
                        "`evaluation_config` settings therein.")
    parser.add_argument("--save-info",
                        default=False,
                        action="store_true",
                        help="Save the info field generated by the step() method, "
                        "as well as the action, observations, rewards and done fields.")
    parser.add_argument("--use-cpu",
                        default=False,
                        action="store_true",
                        help="If provided, uses the cpu instead of cuda.")
    parser.add_argument("--deterministic-policy",
                        default=False,
                        action="store_true",
                        help="If provided, makes the policy deterministic.")
    parser.add_argument("--level-seed",
                        type=int,
                        default=None,
                        help="If provided, only runs on this one level seed.")
    return parser


def get_nrows_ncols(num_plots):
    nrows = int(np.ceil(num_plots / 3))
    ncols = int(np.ceil(num_plots / nrows))
    return nrows, ncols


def visualize_object_masks(info, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    obs = np.array(info["obs"])
    objects = np.squeeze(np.array(info["obj_masks"]))
    nrows, ncols = get_nrows_ncols(objects.shape[1] + 1)
    for i in range(obs.shape[0]):
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 5, nrows * 5))
        axs = np.reshape(axs, -1)
        sample_image = obs[i]

        axs[0].set_xticks([], [])
        axs[0].set_yticks([], [])
        axs[0].imshow(sample_image, aspect="equal")

        sample_objects = objects[i]
        for j in range(sample_objects.shape[0]):
            sample_object = sample_objects[j, :, :]
            axs[j + 1].set_xticks([], [])
            axs[j + 1].set_yticks([], [])
            axs[j + 1].imshow(sample_object, aspect="equal")
        output_filepath = os.path.join(output_dir, f"{i:03d}.png")
        plt.tight_layout()
        plt.savefig(output_filepath)
        plt.close()


def create_video_from_observations(obs, video_filepath, fps=20, max_num_frames=2000):
    obs = np.asarray(obs)
    _, width, height, _ = obs.shape
    video = cv2.VideoWriter(video_filepath, 0, fps, (width, height))
    for image in obs:
        if len(image.shape) == 2 or image.shape[-1] == 3:
            # In this case, the image is the full observation, so just do nothing.
            # Yeah, I know this if statement conditional isn't necessary, it's here for clarity.
            pass
        elif image.shape[-1] > 3:
            # Need to extract the image obs from the stacked (frame-diffed one).
            # Assume the last 3 channels are the actual image.
            image = image[:, :, -3:]

        if image.dtype != np.uint8:
            # Convert to the right datatype.
            # The assumption is that the image is normalized, but it's tough to correctly
            # unnormalize it, so just downcast it and hope for the best.
            image = image.astype(np.uint8)

        video.write(image)

        video.write(image)
    video.release()


def visualize_basic_info(info, output_dir):
    video_filepath = os.path.join(output_dir, "video.avi")
    create_video_from_observations(info["obs"], video_filepath)


def report_info(info):
    print(f"total reward: {np.sum(info['reward'])}")


def visualize_info(info, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report_info(info)
    visualize_basic_info(info, output_dir)
    if "obj_masks" in info:
        visualize_object_masks(info, output_dir)


def run(args, parser):
    config = {}
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    # If no pkl file found, require command line `--config`.
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError("Could not find params.pkl in either the checkpoint dir or "
                             "its parent directory AND no config given on command line!")

    # Load the config from pickled.
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)

    if args.use_cpu:
        # When you don't want to run with any gpus.
        config["num_gpus_per_worker"] = 0
        config["num_gpus"] = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    config["num_workers"] = 1
    # # Set num_workers to be at least 2.
    # if "num_workers" in config:
    #     config["num_workers"] = min(2, config["num_workers"])

    # Merge with `evaluation_config`.
    evaluation_config = copy.deepcopy(config.get("evaluation_config", {}))
    # ADDED
    if args.deterministic_policy:
        evaluation_config["explore"] = False
        config["explore"] = False
    if "env_config" in evaluation_config:
        evaluation_config["env_config"]["num_levels"] = 1
        evaluation_config["env_config"]["use_sequential_levels"] = True
        evaluation_config["env_config"][
            "start_level"] = 0 if args.level_seed is None else args.level_seed
    config["env_config"]["num_levels"] = 1
    config["env_config"]["use_sequential_levels"] = True
    config["env_config"]["start_level"] = 0 if args.level_seed is None else args.level_seed
    # END ADDED
    config = merge_dicts(config, evaluation_config)
    # Merge with command line `--config` settings.
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init()

    # Create the Trainer from config.
    cls = get_trainable_cls(args.run)
    agent = cls(env=args.env, config=config)
    # Load state from checkpoint.
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    num_episodes = int(args.episodes)

    video_dir = None
    if args.video_dir:
        video_dir = os.path.expanduser(args.video_dir)

    vis_info = rollout(agent,
                       args.env,
                       num_steps,
                       num_episodes,
                       video_dir,
                       config,
                       level_seed=args.level_seed)
    visualize_info(vis_info, video_dir)


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""
    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # if num_episodes is set, this overrides num_steps
    if num_episodes:
        return episodes < num_episodes
    # if num_steps is set, continue until we reach the limit
    if num_steps:
        return steps < num_steps
    # otherwise keep going forever
    return True


def get_env(agent, env_name, config, level_seed):
    config["env_config"]["start_level"] = level_seed
    policy_agent_mapping = default_policy_agent_mapping
    if config is None:
        if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
            env = agent.workers.local_worker().env
            multiagent = isinstance(env, MultiAgentEnv)
            if agent.workers.local_worker().multiagent:
                policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]

            policy_map = agent.workers.local_worker().policy_map
            state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
            use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        else:
            env = gym.make(env_name)
            multiagent = False
            try:
                policy_map = {DEFAULT_POLICY_ID: agent.policy}
            except AttributeError:
                raise AttributeError("Agent ({}) does not have a `policy` property! This is needed "
                                     "for performing (trained) agent rollouts.".format(agent))
            use_lstm = {DEFAULT_POLICY_ID: False}
            state_init = None
    else:
        print("attempting to create the env")
        import envs.custom_procgen_env_wrapper
        env = envs.custom_procgen_env_wrapper.create_env(config["env_config"])
        assert hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet)
        multiagent = isinstance(env, MultiAgentEnv)
        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    return env, multiagent, policy_map, use_lstm, state_init


def rollout(agent,
            env_name,
            num_steps,
            num_episodes=0,
            video_dir=None,
            config=None,
            level_seed=None):
    policy_agent_mapping = default_policy_agent_mapping
    env, multiagent, policy_map, use_lstm, state_init = get_env(
        agent,
        env_name,
        config,
        level_seed=0 if level_seed is None else level_seed,
    )
    action_init = {p: _flatten_action(m.action_space.sample()) for p, m in policy_map.items()}

    vis_info = collections.defaultdict(list)
    steps = 0
    episodes = 0
    all_ep_total_reward = 0
    seeds = []
    while keep_going(steps, num_steps, episodes, num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        env, multiagent, policy_map, use_lstm, _ = get_env(
            agent,
            env_name,
            config,
            level_seed=episodes if level_seed is None else level_seed,
        )
        obs = env.reset()
        agent_states = DefaultMapping(lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        episode_steps = 0
        while not done and keep_going(steps, num_steps, episodes, num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(a_obs,
                                                        prev_action=prev_actions[agent_id],
                                                        prev_reward=prev_rewards[agent_id],
                                                        policy_id=policy_id)
                    a_action = _flatten_action(a_action)  # tuple actions
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)

            if done:
                seeds.append(info["level_seed"])
                print(seeds)

            if hasattr(agent.workers.local_worker().get_policy().model, "object_masks"):
                obj_masks = agent.workers.local_worker().get_policy().model.object_masks().cpu(
                ).numpy()
                vis_info["obj_masks"].append(obj_masks)
            vis_info["obs"].append(obs)
            vis_info["reward"].append(reward)

            episode_steps += 1
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            env.render()
            steps += 1
            obs = next_obs
        print("Episode #{}: reward: {} steps: {}".format(episodes, reward_total, episode_steps))
        all_ep_total_reward += reward_total
        if done:
            episodes += 1

    print(f"Average episode reward: {all_ep_total_reward / episodes:.4f}")
    return vis_info


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
