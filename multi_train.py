import argparse
import collections
import copy
import itertools
import os
import yaml

import procgen

from train import run, create_parser


def validate_env_names(env_names):
    for env_name in env_names:
        assert env_name in procgen.env.ENV_NAMES, f"Invalid environment: {env_name}"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_names",
                        type=str,
                        nargs="+",
                        default=procgen.env.ENV_NAMES,
                        help="List of environments to use in the sweep.")
    parser.add_argument("--base_exp_filepath",
                        type=str,
                        default="./experiments/episode_adversarial.yaml",
                        help="Default experiment file to use.")
    parser.add_argument("--num_iterations",
                        type=int,
                        default=1,
                        help="Number of times to run the full sequence of experiments")
    return parser


def determine_algorithm(exp):
    algo = exp["run"]
    if "dqn" in algo.lower():
        return "dqn"
    elif "ppo" in algo.lower():
        return "ppo"
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def named_product(**items):
    Product = collections.namedtuple("Product", items.keys())
    return itertools.starmap(Product, itertools.product(*items.values()))


def sample_ppo_configs(
    base_config,
    learning_rate_options=[0.0005],
    num_sgd_iter_options=[2],
    sgd_minibatch_size_num_filters_pair_options=[
        (1628, [32, 64, 64]),
    ],
    num_envs_rollout_len_pair_options=[(74, 55)],
    frame_stack_k_options=[2],
    transforms_options=[["random_translate"]],
    entropy_coeff_schedule_options=[
        [[0, 0.01], [4000000, 0.005]],
    ],
):
    parameter_settings = named_product(
        learning_rate=learning_rate_options,
        num_sgd_iter=num_sgd_iter_options,
        sgd_minibatch_size_num_filters_pair=sgd_minibatch_size_num_filters_pair_options,
        num_envs_rollout_len_pair=num_envs_rollout_len_pair_options,
        frame_stack_k=frame_stack_k_options,
        transforms=transforms_options,
        entropy_coeff_schedule=entropy_coeff_schedule_options,
    )
    configs = dict()
    for params in parameter_settings:
        config = copy.deepcopy(base_config)

        # Algorithm parameters.
        config["config"]["lr"] = params.learning_rate
        config["config"]["num_sgd_iter"] = params.num_sgd_iter
        config["config"]["sgd_minibatch_size"] = params.sgd_minibatch_size_num_filters_pair[0]
        config["config"]["num_envs_per_worker"] = params.num_envs_rollout_len_pair[0]
        config["config"]["rollout_fragment_length"] = params.num_envs_rollout_len_pair[1]
        config["config"]["entropy_coeff_schedule"] = params.entropy_coeff_schedule

        # Environment parameters.
        env_wrapper_options = {
            "frame_stack": True,
            "frame_stack_options": {
                "k": params.frame_stack_k
            },
            "normalize_reward": False,
            "grayscale": False,
            "mixed_grayscale_color": False,
            "mixed_grayscale_color_options": {
                "num_prev_frames": 1
            }
        }
        config["config"]["env_config"]["env_wrapper_options"] = env_wrapper_options

        # Model parameters.
        custom_model_options = {
            "num_filters": params.sgd_minibatch_size_num_filters_pair[1],
            "data_augmentation_options": {
                "mode": "drac" if len(params.transforms) > 0 else "none",
                "augmentation_mode": "independent",
                "mode_options": {
                    "drac": {
                        "drac_weight": 0.1,
                        "drac_value_weight": 1,
                        "drac_policy_weight": 1,
                    }
                },
                "transforms": params.transforms,
            },
            "dropout_prob": 0.05,
            "optimizer_options": {
                "opt_type": "adam",
                "weight_decay": 0.0,
            },
            "prev_action_mode": "concat",
            "intrinsic_reward_options": {
                "use_noop_penalty": True,
                "noop_penalty_options": {
                    "reward": -0.1
                }
            }
        }
        config["config"]["model"]["custom_options"] = custom_model_options

        exp_name = "_".join([
            "ppo_{}_itr_{}",
            f"lr_{params.learning_rate}",
            f"num_sgd_iter_{params.num_sgd_iter}",
            f"sgd_minibatch_size_{params.sgd_minibatch_size_num_filters_pair[0]}",
            f"num_envs_{params.num_envs_rollout_len_pair[0]}_rollout_length_{params.num_envs_rollout_len_pair[1]}",
            "_".join(params.transforms),
            f"ent_sch_{'_'.join(str(v) for y in params.entropy_coeff_schedule for v in y)}",
            f"num_filters_{'_'.join(str(v) for v in params.sgd_minibatch_size_num_filters_pair[1])}",
        ])
        configs[exp_name] = config

    return configs


def sample_dqn_configs(
    base_config,
    learning_rate_options=[0.0005],
    train_batch_size_num_filters_pair_options=[
        (1200, [24, 48, 48]),
    ],
    num_envs_rollout_len_pair_options=[(512, 64)],
    frame_stack_k_options=[2],
):
    parameter_settings = named_product(
        learning_rate=learning_rate_options,
        train_batch_size_num_filters_pair=train_batch_size_num_filters_pair_options,
        num_envs_rollout_len_pair=num_envs_rollout_len_pair_options,
        frame_stack_k=frame_stack_k_options,
    )
    configs = dict()
    for params in parameter_settings:
        config = copy.deepcopy(base_config)

        # Algorithm parameters.
        config["config"]["lr"] = params.learning_rate
        config["config"]["train_batch_size"] = params.train_batch_size_num_filters_pair[0]
        config["config"]["num_envs_per_worker"] = params.num_envs_rollout_len_pair[0]
        config["config"]["rollout_fragment_length"] = params.num_envs_rollout_len_pair[1]

        # Environment parameters.
        env_wrapper_options = {
            "frame_stack": True,
            "frame_stack_options": {
                "k": params.frame_stack_k
            },
            "normalize_reward": False,
            "grayscale": False,
            "mixed_grayscale_color": False,
            "mixed_grayscale_color_options": {
                "num_prev_frames": 1
            }
        }
        config["config"]["env_config"]["env_wrapper_options"] = env_wrapper_options

        exp_name = "_".join([
            "dqn_10_{}_itr_{}",
            f"lr_{params.learning_rate}",
            f"train_batch_size_{params.train_batch_size_num_filters_pair[0]}",
            f"num_envs_{params.num_envs_rollout_len_pair[0]}_rollout_length_{params.num_envs_rollout_len_pair[1]}",
        ])
        configs[exp_name] = config

    return configs


def sample_configs(algo, base_config, *args, **kwargs):
    if algo == "dqn":
        return sample_dqn_configs(base_config, *args, **kwargs)
    elif algo == "ppo":
        return sample_ppo_configs(base_config, *args, **kwargs)
    else:
        raise ValueError(f"Invalid algorithm: {algo}")


EASY_GAME_RANGES = {
    'coinrun': [0, 5, 10],
    'starpilot': [0, 2.5, 64],
    'caveflyer': [0, 3.5, 12],
    'dodgeball': [0, 1.5, 19],
    'fruitbot': [-12, -1.5, 32.4],
    'chaser': [0, .5, 13],
    'miner': [0, 1.5, 13],
    'jumper': [0, 1, 10],
    'leaper': [0, 1.5, 10],
    'maze': [0, 5, 10],
    'bigfish': [0, 1, 40],
    'heist': [0, 3.5, 10],
    'climber': [0, 2, 12.6],
    'plunder': [0, 4.5, 30],
    'ninja': [0, 3.5, 10],
    'bossfight': [0, .5, 13],
    'caterpillar': [0, 8.25, 24],
    'gemjourney': [0, 1.1, 16],
    'hovercraft': [0, 0.2, 18],
    'safezone': [0, 0.2, 10],
}


def adapt_config_to_env(config, algo, env_dir, env_name):
    config["local_dir"] = env_dir
    config["config"]["env_config"]["env_name"] = env_name
    if algo == "dqn":
        assert env_name in EASY_GAME_RANGES
        config["config"]["v_min"] = EASY_GAME_RANGES[env_name][0]
        config["config"]["v_max"] = EASY_GAME_RANGES[env_name][2]

    return config


def write_experiments(base, algo, num_iterations, env_names):
    exps = dict()
    configs = sample_configs(algo, copy.deepcopy(base))
    for env_name in env_names:
        env_dir = os.path.join(base["local_dir"], env_name)
        for exp_name_template, config in configs.items():
            config = adapt_config_to_env(copy.deepcopy(config), algo, env_dir, env_name)
            for iteration in range(num_iterations):
                exp_name = exp_name_template.format(env_name, iteration)
                exps[exp_name] = config
    os.makedirs(base["local_dir"], exist_ok=True)
    exps_filepath = os.path.join(base["local_dir"], "experiments.yaml")
    with open(exps_filepath, "w", encoding="utf-8") as outfile:
        yaml.dump(exps, outfile, default_flow_style=False)
    return exps_filepath


def run_experiments(experiments_filepath):
    parser = create_parser()
    args = parser.parse_args(args=[])
    args.config_file = experiments_filepath
    run(args, parser)


def main():
    parser = get_parser()
    args = parser.parse_args()
    validate_env_names(args.env_names)
    with open(args.base_exp_filepath) as f:
        base_exp = list(yaml.safe_load(f).values())[0]

    algo = determine_algorithm(base_exp)
    exps_filepath = write_experiments(base_exp, algo, args.num_iterations, args.env_names)
    run_experiments(exps_filepath)


if __name__ == "__main__":
    main()
