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


def named_product(**items):
    Product = collections.namedtuple("Product", items.keys())
    return itertools.starmap(Product, itertools.product(*items.values()))


def sample_configs(
    base_config,
    learning_rate_schedule_options=[
        [[0, 0.0005]],
    ],
    num_sgd_iter_options=[2],
    sgd_minibatch_size_num_filters_pair_options=[
        (814, [32, 64, 64]),
    ],
    num_envs_rollout_len_pair_options=[(74, 55)],
    frame_stack_k_options=[2],
    transforms_options=[["random_translate"]],
    entropy_coeff_schedule_options=[
        [[0, 0.01]],
    ],
    gae_lambda_options=[0.95],
    dropout_prob_options=[0.05],
    drac_weight_options=[0.1],
):
    parameter_settings = named_product(
        learning_rate_schedule=learning_rate_schedule_options,
        num_sgd_iter=num_sgd_iter_options,
        sgd_minibatch_size_num_filters_pair=sgd_minibatch_size_num_filters_pair_options,
        num_envs_rollout_len_pair=num_envs_rollout_len_pair_options,
        frame_stack_k=frame_stack_k_options,
        transforms=transforms_options,
        entropy_coeff_schedule=entropy_coeff_schedule_options,
        gae_lambda=gae_lambda_options,
        dropout_prob=dropout_prob_options,
        drac_weight=drac_weight_options,
    )
    configs = dict()
    for params in parameter_settings:
        config = copy.deepcopy(base_config)

        # Algorithm parameters.
        config["config"]["lr_schedule"] = params.learning_rate_schedule
        config["config"]["num_sgd_iter"] = params.num_sgd_iter
        config["config"]["sgd_minibatch_size"] = params.sgd_minibatch_size_num_filters_pair[0]
        config["config"]["num_envs_per_worker"] = params.num_envs_rollout_len_pair[0]
        config["config"]["rollout_fragment_length"] = params.num_envs_rollout_len_pair[1]
        config["config"]["entropy_coeff_schedule"] = params.entropy_coeff_schedule
        config["config"]["lambda"] = params.gae_lambda

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
                        "drac_weight": params.drac_weight,
                        "drac_value_weight": 1,
                        "drac_policy_weight": 1,
                    }
                },
                "transforms": params.transforms,
            },
            "dropout_prob": params.dropout_prob,
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
            "{}_itr_{}",
            f"lr_sch_{'_'.join(str(v) for y in params.learning_rate_schedule for v in y)}",
            f"num_sgd_iter_{params.num_sgd_iter}",
            f"sgd_minibatch_size_{params.sgd_minibatch_size_num_filters_pair[0]}",
            f"num_envs_{params.num_envs_rollout_len_pair[0]}_rollout_length_{params.num_envs_rollout_len_pair[1]}",
            "_".join(params.transforms),
            f"ent_sch_{'_'.join(str(v) for y in params.entropy_coeff_schedule for v in y)}",
            f"gae_lambda_{params.gae_lambda}",
            f"num_filters_{'_'.join(str(v) for v in params.sgd_minibatch_size_num_filters_pair[1])}",
            f"dropout_{params.dropout_prob}",
            f"drac_weight_{params.drac_weight}",
        ])
        configs[exp_name] = config

    return configs


def write_experiments(base, num_iterations, env_names):
    exps = dict()
    # start manual stuff :(
    configs = sample_configs(copy.deepcopy(base),
                             learning_rate_schedule_options=[
                                 [[0, 5.0e-4], [4000000, 5.0e-4], [4000001, 2.5e-4]],
                             ])
    configs.update(sample_configs(copy.deepcopy(base), dropout_prob_options=[0.1]))
    configs.update(sample_configs(copy.deepcopy(base), drac_weight_options=[0.2]))
    # end manual stuff
    for env_name in env_names:
        env_dir = os.path.join(base["local_dir"], env_name)
        for exp_name_template, config in configs.items():
            config = copy.deepcopy(config)
            config["local_dir"] = env_dir
            config["config"]["env_config"]["env_name"] = env_name
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
    if args.env_names == ["all"]:
        args.env_names = list(procgen.env.ENV_NAMES)
    validate_env_names(args.env_names)
    with open(args.base_exp_filepath) as f:
        base_exp = list(yaml.safe_load(f).values())[0]
    exps_filepath = write_experiments(base_exp, args.num_iterations, args.env_names)
    run_experiments(exps_filepath)


if __name__ == "__main__":
    main()
