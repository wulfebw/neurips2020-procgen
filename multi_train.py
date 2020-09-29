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
            # This information is common to all the experiments.
            base_copy = copy.deepcopy(base)
            base_copy["local_dir"] = env_dir
            base_copy["config"]["env_config"]["env_name"] = env_name

            for transforms in [
                ["random_translate"],
            ]:
                for data_aug_mode in ["drac"]:
                    if len(transforms) == 0:
                        data_aug_mode = "none"
                    for lr in [0.0005]:
                        for weight_decay in [0.0]:
                            for drac_policy_weight in [1]:
                                # This information is common to all the experiments.
                                base_copy = copy.deepcopy(base)
                                base_copy["local_dir"] = env_dir
                                base_copy["config"]["env_config"]["env_name"] = env_name

                                base_copy["config"]["lr"] = lr

                                # Env options.
                                env_wrapper_options = {
                                    "frame_stack": True,
                                    "frame_stack_options": {
                                        "k": 2
                                    },
                                    "normalize_reward": False,
                                    "grayscale": False,
                                    "mixed_grayscale_color": False,
                                    "mixed_grayscale_color_options": {
                                        "num_prev_frames": 1
                                    }
                                }
                                base_copy["config"]["env_config"]["env_wrapper_options"].update(
                                    env_wrapper_options)

                                # Random translate versus baseline.
                                custom_model_options = {
                                    "num_filters": [16, 32, 32],
                                    "data_augmentation_options": {
                                        "mode": data_aug_mode,
                                        "augmentation_mode": "independent",
                                        "mode_options": {
                                            "drac": {
                                                "drac_weight": 0.1,
                                                "drac_value_weight": 1,
                                                "drac_policy_weight": drac_policy_weight,
                                            }
                                        },
                                        "transforms": transforms,
                                    },
                                    "dropout_prob": 0.05,
                                    "optimizer_options": {
                                        "opt_type": "adam",
                                        "weight_decay": weight_decay,
                                    },
                                    "prev_action_mode": "concat"
                                }
                                base_copy["config"]["model"][
                                    "custom_options"] = custom_model_options
                                transform_string = "_".join(transforms)

                                exp_name = (
                                    f"itr_{iteration}_{env_name}_{data_aug_mode}_"
                                    f"transforms_{transform_string}_concat_action_002")
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
