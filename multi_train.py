import argparse
import collections.abc
import copy
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


def write_experiments(base, num_iterations, env_names):
    exps = dict()
    for env_name in env_names:
        env_dir = os.path.join(base["local_dir"], env_name)
        for iteration in range(num_iterations):
            for frame_diff in [True, False]:
                # for ep_adv_weight in [0, 1]:

                base_copy = copy.deepcopy(base)
                base_copy["local_dir"] = env_dir
                base_copy["config"]["env_config"]["env_name"] = env_name

                # base_copy["config"]["model"]["custom_options"][
                #     "discriminator_weight"] = ep_adv_weight
                # exp_name = f"itr_{iteration}_{env_name}_ep_adv_{ep_adv_weight}"

                if frame_diff:
                    env_wrapper_options = {
                        "frame_diff": True,
                        "frame_diff_options": {
                            "grayscale": False,
                            "dt": 2
                        },
                        "normalize_obs": True
                    }
                    custom_model_options = {
                        "discriminator_weight": 0,
                        "l2_weight": 0.0001,
                        "late_fusion": True
                    }
                else:
                    env_wrapper_options = {
                        "frame_diff": False,
                        "frame_diff_options": {
                            "grayscale": False,
                            "dt": 2
                        },
                        "normalize_obs": False
                    }
                    custom_model_options = {
                        "discriminator_weight": 0,
                        "l2_weight": 0.0001,
                        "late_fusion": False
                    }

                base_copy["config"]["num_workers"] = 0

                base_copy["config"]["env_config"]["env_wrapper_options"] = env_wrapper_options
                base_copy["config"]["model"]["custom_options"] = custom_model_options
                exp_name = f"itr_{iteration}_{env_name}_frame_diff_{frame_diff}"

                exps[exp_name] = base_copy

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
    exps_filepath = write_experiments(base_exp, args.num_iterations, args.env_names)
    run_experiments(exps_filepath)


if __name__ == "__main__":
    main()
