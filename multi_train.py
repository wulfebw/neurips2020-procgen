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


class PPOSamplingParams:
    def __init__(self, num_workers, num_envs_per_worker, rollout_fragment_length,
                 sgd_minibatch_size):
        self.num_workers = num_workers
        self.num_envs_per_worker = num_envs_per_worker
        self.rollout_fragment_length = rollout_fragment_length
        self.sgd_minibatch_size = sgd_minibatch_size

    def __repr__(self):
        return "_".join([
            f"num_work_{self.num_workers}",
            f"num_envs_per_{self.num_envs_per_worker}",
            f"rollout_len_{self.rollout_fragment_length}",
            f"minibatch_{self.sgd_minibatch_size}",
        ])

    @property
    def num_gpus_per_worker(self):
        return 0.01

    @property
    def num_gpus(self):
        return 1 - (self.num_workers * self.num_gpus_per_worker)


def get_is_recurrent(config):
    if "rnn" in config["config"]["model"]["custom_model"]:
        return True
    return False


def set_env_params(config, params, is_recurrent):
    # Environment parameters.
    use_frame_stack = True
    if is_recurrent or params.frame_stack_k == 1:
        use_frame_stack = False

    env_wrapper_options = {
        "frame_stack": use_frame_stack,
        "frame_stack_options": {
            "k": params.frame_stack_k
        },
        "frame_diff": False,
        "normalize_reward": True if params.reward_normalization_mode == "env_rew_norm" else False,
        "normalize_reward_options": {
            "discount": config["config"]["gamma"]
        },
        "grayscale": False,
        "mixed_grayscale_color": False,
        "mixed_grayscale_color_options": {
            "num_prev_frames": 1
        },
        "frame_stack_phase_correlation": False,
    }
    config["config"]["env_config"]["env_wrapper_options"] = env_wrapper_options
    return config


def set_ppo_algorithm_params(config, params):
    config["config"]["lr"] = params.learning_rate
    config["config"]["num_sgd_iter"] = params.num_sgd_iter
    config["config"]["sgd_minibatch_size"] = params.sampling_params.sgd_minibatch_size
    config["config"]["num_workers"] = params.sampling_params.num_workers
    config["config"]["num_envs_per_worker"] = params.sampling_params.num_envs_per_worker
    config["config"]["rollout_fragment_length"] = params.sampling_params.rollout_fragment_length
    config["config"]["entropy_coeff_schedule"] = params.entropy_coeff_schedule
    config["config"]["reward_normalization_options"] = {"mode": params.reward_normalization_mode}
    config["config"]["grad_clip"] = params.grad_clip_pair[0]
    config["config"]["grad_clip_elementwise"] = params.grad_clip_pair[1]

    # All that matters is these gpu resource parameters add to 1.
    config["config"]["num_gpus_per_worker"] = params.sampling_params.num_gpus_per_worker
    config["config"]["num_gpus"] = params.sampling_params.num_gpus
    return config


def set_ppo_model_params(config, params, is_recurrent):
    if is_recurrent:
        config["config"]["model"]["max_seq_len"] = params.max_seq_len
        config["config"]["model"]["lstm_cell_size"] = params.lstm_cell_size
        config["config"]["model"]["use_lstm"] = True

    # Params common to cnn and rnn.
    custom_model_options = {
        "num_filters": params.num_filters,
        "dropout_prob": params.dropout_prob,
        "prev_action_mode": "concat",
        "weight_init": params.weight_init,
        "data_augmentation_options": {
            "mode": "drac" if len(params.transforms) > 0 else "none",
            "augmentation_mode": "independent",
            "mode_options": {
                "drac": {
                    "drac_weight": params.drac_weight,
                    "drac_value_weight": 1,
                    "drac_policy_weight": 1,
                    "recurrent_repeat_transform": True,
                }
            },
            "transforms": params.transforms,
        },
        "optimizer_options": {
            "opt_type": "adam",
            "weight_decay": 0.0,
        },
        "intrinsic_reward_options": {
            "use_noop_penalty": True,
            "noop_penalty_options": {
                "reward": -0.1
            }
        },
        "fc_activation": params.fc_activation,
        "fc_size": params.fc_size,
    }
    config["config"]["model"]["custom_options"] = custom_model_options
    return config


def get_ppo_exp_name(params, is_recurrent):
    # Start with the common params.
    exp_name = "_".join([
        "{}_itr_{}",
        f"{'rnn' if is_recurrent else 'cnn'}",
        f"lr_{params.learning_rate}",
        f"sgd_itr_{params.num_sgd_iter}",
        f"filters_{'_'.join(str(v) for v in params.num_filters)}",
        f"fc_size_{params.fc_size}",
        f"{params.sampling_params}",
        "_".join(params.transforms),
        f"ent_sch_{'_'.join(str(v) for y in params.entropy_coeff_schedule for v in y)}",
        f"dropout_{params.dropout_prob}",
        f"drac_{params.drac_weight}",
        f"grad_clip_{params.grad_clip_pair[0]}_{params.grad_clip_pair[1]}",
        f"act_fn_{params.fc_activation}",
        f"weight_init_{params.weight_init}",
    ])

    # Add the params that only apply in the recurrent or non-recurrent cases.
    if is_recurrent:
        exp_name += "_" + "_".join([
            f"rnn_size_{params.lstm_cell_size}",
            f"weight_init_{params.weight_init}",
            f"max_seq_len_{params.max_seq_len}",
        ])
    else:
        exp_name += "_" + "_".join([
            f"frame_stack_{params.frame_stack_k}",
        ])
    return exp_name


def sample_configs(
    base_config,
    learning_rate_options=[0.0005],
    num_sgd_iter_options=[2],
    num_filters_options=[[32, 48, 64]],
    fc_size_options=[256],
    lstm_cell_size_options=[256],
    weight_init_options=["default"],
    sampling_params_options=[PPOSamplingParams(4, 64, 64, 1024)],
    max_seq_len_options=[16],
    reward_normalization_mode_options=["running_return"],
    frame_stack_k_options=[2],
    transforms_options=[["random_translate"]],
    entropy_coeff_schedule_options=[
        [[0, 0.01]],
    ],
    fc_activation_options=["relu"],
    grad_clip_pair_options=[(10, 1.0)],
    dropout_prob_options=[0.1],
    drac_weight_options=[0.1],
):
    is_recurrent = get_is_recurrent(base_config)
    parameter_settings = named_product(
        learning_rate=learning_rate_options,
        num_sgd_iter=num_sgd_iter_options,
        num_filters=num_filters_options,
        fc_size=fc_size_options,
        lstm_cell_size=lstm_cell_size_options,
        weight_init=weight_init_options,
        sampling_params=sampling_params_options,
        max_seq_len=max_seq_len_options,
        reward_normalization_mode=reward_normalization_mode_options,
        frame_stack_k=frame_stack_k_options,
        transforms=transforms_options,
        entropy_coeff_schedule=entropy_coeff_schedule_options,
        fc_activation=fc_activation_options,
        grad_clip_pair=grad_clip_pair_options,
        dropout_prob=dropout_prob_options,
        drac_weight=drac_weight_options,
    )
    configs = dict()
    for params in parameter_settings:
        config = copy.deepcopy(base_config)
        config = set_ppo_algorithm_params(config, params)
        config = set_env_params(config, params, is_recurrent)
        config = set_ppo_model_params(config, params, is_recurrent)
        exp_name = get_ppo_exp_name(params, is_recurrent)
        configs[exp_name] = config
    return configs


def write_experiments(base, num_iterations, env_names):
    exps = dict()
    configs = sample_configs(copy.deepcopy(base))
    configs.update(sample_configs(copy.deepcopy(base), frame_stack_k_options=[3]))
    configs.update(sample_configs(copy.deepcopy(base), weight_init_options=["orthogonal"]))
    configs.update(sample_configs(copy.deepcopy(base), fc_activation_options=["tanh"]))
    configs.update(sample_configs(copy.deepcopy(base), grad_clip_pair_options=[(1.0, 1.0)]))
    configs.update(sample_configs(copy.deepcopy(base), fc_size_options=[512]))
    configs.update(
        sample_configs(copy.deepcopy(base),
                       sampling_params_options=[PPOSamplingParams(4, 128, 32, 1024)]))
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
    validate_env_names(args.env_names)
    with open(args.base_exp_filepath) as f:
        base_exp = list(yaml.safe_load(f).values())[0]
    exps_filepath = write_experiments(base_exp, args.num_iterations, args.env_names)
    run_experiments(exps_filepath)


if __name__ == "__main__":
    main()
