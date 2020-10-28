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
    def __init__(self,
                 num_workers,
                 num_envs_per_worker,
                 rollout_fragment_length,
                 sgd_minibatch_size,
                 num_sgd_iter=2):
        self.num_workers = num_workers
        self.num_envs_per_worker = num_envs_per_worker
        self.rollout_fragment_length = rollout_fragment_length
        self.sgd_minibatch_size = sgd_minibatch_size
        self.num_sgd_iter = num_sgd_iter

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

    @property
    def train_batch_size(self):
        return (self.num_workers * self.num_envs_per_worker * self.rollout_fragment_length)

    @property
    def total_minibatches_per_train_iteration(self):
        return int((self.train_batch_size / self.sgd_minibatch_size) * self.num_sgd_iter)


class PPOGradClipParams:
    def __init__(self, grad_clip_elementwise, mode, clip_max, clip_min, percentile, buffer_size):
        """Represents the grad clipping params.


        `clip_max` is used as the default grad clip when running in constant mode.
        """
        assert grad_clip_elementwise > 0
        self.grad_clip_elementwise = grad_clip_elementwise
        assert mode in ["constant", "adaptive"]
        self.mode = mode
        assert clip_max > 0
        self.clip_max = clip_max
        assert clip_min > 0 and clip_min < clip_max
        self.clip_min = clip_min
        assert percentile >= 0 and percentile <= 100
        self.percentile = percentile
        assert buffer_size > 0
        self.buffer_size = buffer_size

    def options(self):
        return {
            "mode": self.mode,
            "adaptive_max": self.clip_max,
            "adaptive_min": self.clip_min,
            "adaptive_percentile": self.percentile,
            "adaptive_buffer_size": self.buffer_size,
        }

    @property
    def grad_clip(self):
        return self.clip_max

    def __repr__(self):
        if self.mode == "constant":
            return f"constant_{self.grad_clip}_{self.grad_clip_elementwise}"
        elif self.mode == "adaptive":
            return "adaptive_" + "_".join(
                str(v) for v in [
                    self.clip_max,
                    self.clip_min,
                    self.percentile,
                    self.buffer_size,
                ])
        else:
            raise ValueError("Unsupported mode: {self.mode}")


class IntrinsicRewardParams:
    def __init__(
        self,
        use_noop_penalty=False,
        noop_penalty_options={"reward": -0.1},
        use_state_revisitation_penalty=False,
        state_revisitation_penalty_options={"reward": -0.01},
    ):
        self.use_noop_penalty = use_noop_penalty
        self.noop_penalty_options = noop_penalty_options
        self.use_state_revisitation_penalty = use_state_revisitation_penalty
        self.state_revisitation_penalty_options = state_revisitation_penalty_options

    def options(self):
        return {
            "use_noop_penalty": self.use_noop_penalty,
            "noop_penalty_options": self.noop_penalty_options,
            "use_state_revisitation_penalty": self.use_state_revisitation_penalty,
            "state_revisitation_penalty_options": self.state_revisitation_penalty_options,
        }

    def __repr__(self):
        rep = "intrins_reward"
        if self.use_noop_penalty:
            rep += "_noop"
        if self.use_state_revisitation_penalty:
            rep += "_revisit"
        return rep


class AutoDracParams:
    def __init__(
        self,
        active=False,
        always_use_transforms=[],
        choose_between_transforms=[
            "random_translate",
            "random_rotation",
            "random_flip_left_right",
            "random_flip_up_down",
        ],
        learner_class="ucb",
        ucb_options={
            "num_steps_per_update": None,
            "q_alpha": 0.01,
            "mean_reward_alpha": 0.05,
            "lmbda": 0.25,
            "ucb_c": 0.01,
            "internal_reward_mode": "return",
        }):
        self.active = active
        self.always_use_transforms = always_use_transforms
        self.choose_between_transforms = choose_between_transforms
        self.learner_class = learner_class
        self.ucb_options = ucb_options

    def options(self, num_steps_per_update):
        ucb_options = copy.deepcopy(self.ucb_options)
        ucb_options["num_steps_per_update"] = num_steps_per_update
        return {
            "active": self.active,
            "always_use_transforms": self.always_use_transforms,
            "choose_between_transforms": self.choose_between_transforms,
            "learner_class": self.learner_class,
            "ucb_options": ucb_options,
        }

    def __repr__(self):
        if not self.active:
            return "False"
        return "_".join([
            f"{self.active}",
            f"{self.ucb_options['q_alpha']}",
            f"{self.ucb_options['lmbda']}",
            f"{self.ucb_options['ucb_c']}",
            f"{self.ucb_options['internal_reward_mode']}",
        ])


class PhasicParams:
    def __init__(self,
                 active=True,
                 aux_loss_every_k=32,
                 aux_loss_num_sgd_iter=4,
                 use_data_aug=True,
                 policy_loss_mode="simple",
                 aux_loss_start_after_num_steps=0,
                 detach_value_head=False):
        self.active = active
        self.aux_loss_every_k = aux_loss_every_k
        self.aux_loss_num_sgd_iter = aux_loss_num_sgd_iter
        self.use_data_aug = use_data_aug
        self.policy_loss_mode = policy_loss_mode
        self.aux_loss_start_after_num_steps = aux_loss_start_after_num_steps
        self.detach_value_head = detach_value_head

    def options(self):
        return dict(
            use_data_aug=self.use_data_aug,
            policy_loss_mode=self.policy_loss_mode,
            detach_value_head=self.detach_value_head,
        )

    def __repr__(self):
        if not self.active:
            return ""

        rep = "_".join([
            f"phasic_{self.aux_loss_every_k}",
            f"{self.aux_loss_num_sgd_iter}",
            f"{self.policy_loss_mode}",
            f"{self.aux_loss_start_after_num_steps}",
            f"detach_{self.detach_value_head}",
        ])
        if self.use_data_aug:
            rep += "_w_data_aug"
        return rep


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
        "normalize_reward": params.reward_normalization_params["mode"] == "env_rew_norm",
        "normalize_reward_options": {
            "discount": config["config"]["gamma"]
        },
        "grayscale": False,
        "mixed_grayscale_color": False,
        "mixed_grayscale_color_options": {
            "num_prev_frames": 1
        },
        "frame_stack_phase_correlation": False,
        "count_state_occupancy": params.intrinsic_reward_params.use_state_revisitation_penalty,
    }
    config["config"]["env_config"]["env_wrapper_options"] = env_wrapper_options
    return config


def set_ppo_algorithm_params(config, params):
    config["config"]["lr"] = params.learning_rate
    config["config"]["lr_schedule"] = params.learning_rate_schedule
    config["config"]["num_sgd_iter"] = params.num_sgd_iter
    config["config"]["sgd_minibatch_size"] = params.sampling_params.sgd_minibatch_size
    config["config"]["num_workers"] = params.sampling_params.num_workers
    config["config"]["num_envs_per_worker"] = params.sampling_params.num_envs_per_worker
    config["config"]["rollout_fragment_length"] = params.sampling_params.rollout_fragment_length
    config["config"]["train_batch_size"] = params.sampling_params.train_batch_size
    config["config"]["vf_loss_coeff"] = params.vf_loss_coeff
    config["config"]["entropy_coeff_schedule"] = params.entropy_coeff_schedule
    config["config"]["reward_normalization_options"] = params.reward_normalization_params
    config["config"]["grad_clip"] = params.grad_clip_params.grad_clip
    config["config"]["grad_clip_elementwise"] = params.grad_clip_params.grad_clip_elementwise
    config["config"]["grad_clip_options"] = params.grad_clip_params.options()
    config["config"]["auto_drac_options"] = params.auto_drac_params.options(
        params.sampling_params.total_minibatches_per_train_iteration)
    config["config"]["use_phasic_optimizer"] = params.phasic_params.active
    config["config"]["aux_loss_every_k"] = params.phasic_params.aux_loss_every_k
    config["config"]["aux_loss_num_sgd_iter"] = params.phasic_params.aux_loss_num_sgd_iter
    config["config"][
        "aux_loss_start_after_num_steps"] = params.phasic_params.aux_loss_start_after_num_steps

    # All that matters is these gpu resource parameters add to 1.
    config["config"]["num_gpus_per_worker"] = params.sampling_params.num_gpus_per_worker
    config["config"]["num_gpus"] = params.sampling_params.num_gpus
    return config


def get_trainer_mode(params):
    if params.phasic_params.active:
        return "phasic"
    elif len(params.transforms) > 0:
        return "drac"
    else:
        return "none"


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
            "mode": get_trainer_mode(params),
            "augmentation_mode": "independent",
            "mode_options": {
                "drac": {
                    "drac_weight": params.drac_weight,
                    "drac_value_weight": 1,
                    "drac_policy_weight": 1,
                    "recurrent_repeat_transform": True,
                },
                "phasic": params.phasic_params.options(),
            },
            "transforms": params.transforms,
        },
        "optimizer_options": {
            "opt_type": "adam",
            "weight_decay": 0.0,
        },
        "fc_activation": params.fc_activation,
        "fc_size": params.fc_size,
    }
    custom_model_options["intrinsic_reward_options"] = params.intrinsic_reward_params.options()
    config["config"]["model"]["custom_options"] = custom_model_options
    return config


def get_transform_abbreviation(transform):
    if transform == "random_translate":
        return "rt"
    elif transform == "random_flip_up_down":
        return "fup"
    elif transform == "random_flip_left_right":
        return "flp"
    elif transform == "random_rotation":
        return "rr"
    else:
        raise NotImplementedError(f"Transform not implemented {transform}")


def get_ppo_exp_name(params, is_recurrent):
    # Start with the common params.
    if params.learning_rate_schedule is not None:
        lr_str = f"lr_sch_{'_'.join(str(v) for y in params.learning_rate_schedule for v in y)}"
    else:
        lr_str = f"lr_{params.learning_rate}"

    exp_name = "_".join([
        "{}_itr_{}",
        f"{'rnn' if is_recurrent else 'cnn'}",
        f"{lr_str}",
        # f"sgd_itr_{params.num_sgd_iter}",
        f"filters_{'_'.join(str(v) for v in params.num_filters)}",
        # f"fc_size_{params.fc_size}",
        f"{params.sampling_params}",
        "_".join([get_transform_abbreviation(t) for t in params.transforms]),
        f"ent_sch_{'_'.join(str(v) for y in params.entropy_coeff_schedule for v in y)}",
        # f"dropout_{params.dropout_prob}",
        f"vf_loss_coeff_{params.vf_loss_coeff}",
        # f"drac_{params.drac_weight}",
        f"grad_clip_{str(params.grad_clip_params)}",
        # f"act_fn_{params.fc_activation}",
        # f"weight_init_{params.weight_init}",
        f"rew_norm_alpha_{params.reward_normalization_params['alpha']}",
        f"{params.intrinsic_reward_params}",
        f"auto_drac_{params.auto_drac_params}",
        f"{params.phasic_params}",
    ])

    # Add the params that only apply in the recurrent or non-recurrent cases.
    if is_recurrent:
        exp_name += "_" + "_".join([
            f"rnn_size_{params.lstm_cell_size}",
            f"weight_init_{params.weight_init}",
            f"max_seq_len_{params.max_seq_len}",
        ])
    return exp_name


def sample_configs(base_config,
                   learning_rate_options=[0.0005],
                   learning_rate_schedule_options=[None],
                   num_sgd_iter_options=[1],
                   num_filters_options=[[32, 48, 64]],
                   fc_size_options=[256],
                   lstm_cell_size_options=[256],
                   weight_init_options=["default"],
                   sampling_params_options=[PPOSamplingParams(7, 128, 16, 1792)],
                   max_seq_len_options=[16],
                   reward_normalization_params_options=[{
                       "mode": "running_return",
                       "alpha": 0.005,
                   }],
                   frame_stack_k_options=[2],
                   transforms_options=[["random_translate"]],
                   entropy_coeff_schedule_options=[
                       [[0, 0.01]],
                   ],
                   fc_activation_options=["relu"],
                   grad_clip_params_options=[PPOGradClipParams(1.0, "constant", 1.0, 0.1, 95, 128)],
                   dropout_prob_options=[0.1],
                   drac_weight_options=[0.1],
                   intrinsic_reward_params_options=[IntrinsicRewardParams()],
                   auto_drac_params_options=[AutoDracParams()],
                   phasic_params_options=[PhasicParams()],
                   vf_loss_coeff_options=[0.25]):
    is_recurrent = get_is_recurrent(base_config)
    parameter_settings = named_product(
        learning_rate=learning_rate_options,
        learning_rate_schedule=learning_rate_schedule_options,
        num_sgd_iter=num_sgd_iter_options,
        num_filters=num_filters_options,
        fc_size=fc_size_options,
        lstm_cell_size=lstm_cell_size_options,
        weight_init=weight_init_options,
        sampling_params=sampling_params_options,
        max_seq_len=max_seq_len_options,
        reward_normalization_params=reward_normalization_params_options,
        frame_stack_k=frame_stack_k_options,
        transforms=transforms_options,
        entropy_coeff_schedule=entropy_coeff_schedule_options,
        fc_activation=fc_activation_options,
        grad_clip_params=grad_clip_params_options,
        dropout_prob=dropout_prob_options,
        drac_weight=drac_weight_options,
        intrinsic_reward_params=intrinsic_reward_params_options,
        auto_drac_params=auto_drac_params_options,
        phasic_params=phasic_params_options,
        vf_loss_coeff=vf_loss_coeff_options,
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
    configs = dict()
    configs.update(sample_configs(copy.deepcopy(base)))
    configs.update(
        sample_configs(
            copy.deepcopy(base),
            phasic_params_options=[
                PhasicParams(active=True,
                             aux_loss_every_k=32,
                             aux_loss_num_sgd_iter=3,
                             use_data_aug=True,
                             policy_loss_mode="simple",
                             aux_loss_start_after_num_steps=0,
                             detach_value_head=False),
            ],
        ))
    configs.update(
        sample_configs(
            copy.deepcopy(base),
            phasic_params_options=[
                PhasicParams(active=True,
                             aux_loss_every_k=32,
                             aux_loss_num_sgd_iter=6,
                             use_data_aug=True,
                             policy_loss_mode="simple",
                             aux_loss_start_after_num_steps=0,
                             detach_value_head=False),
            ],
        ))
    configs.update(
        sample_configs(
            copy.deepcopy(base),
            phasic_params_options=[
                PhasicParams(active=True,
                             aux_loss_every_k=32,
                             aux_loss_num_sgd_iter=6,
                             use_data_aug=True,
                             policy_loss_mode="simple",
                             aux_loss_start_after_num_steps=0,
                             detach_value_head=False),
            ],
            num_filters_options=[[24, 48, 48]],
        ))
    configs.update(
        sample_configs(
            copy.deepcopy(base),
            grad_clip_params_options=[PPOGradClipParams(0.1, "constant", 1.0, 0.1, 95, 128)],
        ))
    configs.update(
        sample_configs(
            copy.deepcopy(base),
            sampling_params_options=[PPOSamplingParams(7, 64, 32, 1792)],
        ))
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
