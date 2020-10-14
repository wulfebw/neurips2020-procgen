from ray.rllib.agents.dqn.dqn import (DEFAULT_CONFIG, validate_config, get_initial_state,
                                      update_worker_exploration, after_train_result,
                                      update_target_if_needed, execution_plan)
from ray.rllib.agents.dqn.dqn_tf_policy import (QLoss, ComputeTDErrorMixin, build_q_model,
                                                get_distribution_inputs_and_class, build_q_losses,
                                                adam_optimizer, clip_gradients, build_q_stats,
                                                setup_early_mixins, setup_mid_mixins,
                                                setup_late_mixins, compute_q_values, _adjust_nstep,
                                                postprocess_nstep_and_prio)
from ray.rllib.agents.dqn.simple_q_tf_policy import TargetNetworkMixin
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.optimizers import SyncReplayOptimizer
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy


def build_data_augmented_q_losses(policy, model, _, train_batch):
    return build_q_losses(policy, model, _, train_batch)


DataAugmentingDQNTFPolicy = build_tf_policy(
    name="DataAugmentingDQNTFPolicy",
    get_default_config=lambda: DEFAULT_CONFIG,
    make_model=build_q_model,
    action_distribution_fn=get_distribution_inputs_and_class,
    loss_fn=build_data_augmented_q_losses,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    gradients_fn=clip_gradients,
    extra_action_fetches_fn=lambda policy: {"q_values": policy.q_values},
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.q_loss.td_error},
    before_init=setup_early_mixins,
    before_loss_init=setup_mid_mixins,
    after_init=setup_late_mixins,
    obs_include_prev_action_reward=False,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ])


def get_policy_class(config):
    return DataAugmentingDQNTFPolicy


def before_learn_on_batch(samples, policy_map, batch_size):
    import ipdb
    ipdb.set_trace()
    return samples


def make_data_augmenting_policy_optimizer(workers, config):
    kwargs = {"prioritized_replay": config.get("prioritized_replay", False)}
    kwargs.update(**config["optimizer"])
    if "prioritized_replay" in config:
        kwargs.update({
            "prioritized_replay_alpha":
            config["prioritized_replay_alpha"],
            "prioritized_replay_beta":
            config["prioritized_replay_beta"],
            "prioritized_replay_beta_annealing_timesteps":
            config["prioritized_replay_beta_annealing_timesteps"],
            "final_prioritized_replay_beta":
            config["final_prioritized_replay_beta"],
            "prioritized_replay_eps":
            config["prioritized_replay_eps"],
        })
    return SyncReplayOptimizer(workers,
                               learning_starts=config["learning_starts"],
                               buffer_size=config["buffer_size"],
                               train_batch_size=config["train_batch_size"],
                               before_learn_on_batch=before_learn_on_batch,
                               **kwargs)


DataAugmentingDQNTrainer = build_trainer(
    name="data_augmenting_dqn_trainer",
    default_policy=DataAugmentingDQNTFPolicy,
    get_policy_class=get_policy_class,
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    get_initial_state=get_initial_state,
    make_policy_optimizer=make_data_augmenting_policy_optimizer,
    before_train_step=update_worker_exploration,
    after_optimizer_step=update_target_if_needed,
    after_train_result=after_train_result,
)
#   execution_plan=execution_plan)
