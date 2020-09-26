import collections

import numpy as np

from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.ppo.ppo import (DEFAULT_CONFIG, validate_config, update_kl,
                                      warn_about_bad_reward_scales, choose_policy_optimizer)
from ray.rllib.agents.ppo.ppo_tf_policy import postprocess_ppo_gae, setup_config
from ray.rllib.agents.ppo.ppo_torch_policy import (kl_and_loss_stats, vf_preds_fetches,
                                                   setup_mixins, KLCoeffMixin, ValueNetworkMixin)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

from algorithms.data_augmenting_ppo_agent.data_augmenting_ppo_agent import data_augmenting_loss
from algorithms.data_augmenting_ppo_agent.ucb_learner import UCBLearner


def drac_init(policy, observation_space, action_space, options):
    setup_mixins(policy, observation_space, action_space, options)

    # Set up drac variables.
    options = options["model"]["custom_options"]["auto_drac_options"]
    policy.always_use_transforms = options["always_use_transforms"]
    policy.choose_between_transforms = []
    for t in options["choose_between_transforms"]:
        if isinstance(t, list):
            t = tuple(t)
        policy.choose_between_transforms.append(t)
    if options["learner_class"] == "ucb":
        policy.transform_selector = UCBLearner(policy.choose_between_transforms,
                                               **options["ucb_options"],
                                               verbose=True)
    else:
        raise NotImplementedError(f"Learner not implemented: {options.learner_class}")


def drac_loss_fn(policy, model, dist_class, train_batch):
    transform_index, transform_info = policy.transform_selector.step(
        list(train_batch["rewards"].detach().cpu().numpy()))
    policy.transform_info = transform_info

    transform = policy.choose_between_transforms[transform_index]
    if isinstance(transform, str):
        formatted_transform = [transform]
    elif isinstance(transform, tuple):
        formatted_transform = list(transform)
    else:
        raise ValueError(f"Invalid transform: {transform}")

    model.data_augmentation_options["transforms"] = (policy.always_use_transforms +
                                                     formatted_transform)
    return data_augmenting_loss(policy, model, dist_class, train_batch)


def drac_stats_fn(policy, train_batch):
    stats = kl_and_loss_stats(policy, train_batch)

    drac_stats = {}
    drac_stats.update(
        {f"drac_{k}_value": v
         for (k, v) in policy.transform_info["action_values"].items()})
    drac_stats.update(
        {f"drac_{k}_count": v
         for (k, v) in policy.transform_info["action_counts"].items()})
    drac_stats.update(
        {f"drac_{k}_eligibility": v
         for (k, v) in policy.transform_info["eligibility"].items()})
    drac_stats["overall_mean_reward"] = policy.transform_info["overall_mean_reward"]
    stats.update(drac_stats)
    return stats


DRACTorchPolicy = build_torch_policy(name="DRACPolicy",
                                     get_default_config=lambda: DEFAULT_CONFIG,
                                     loss_fn=drac_loss_fn,
                                     stats_fn=drac_stats_fn,
                                     extra_action_out_fn=vf_preds_fetches,
                                     postprocess_fn=postprocess_ppo_gae,
                                     extra_grad_process_fn=apply_grad_clipping,
                                     before_init=setup_config,
                                     after_init=drac_init,
                                     mixins=[KLCoeffMixin, ValueNetworkMixin])


def get_policy_class(config):
    return DRACTorchPolicy


DRACPPOTrainer = build_trainer(name="data_augmenting_ppo_trainer",
                               default_config=DEFAULT_CONFIG,
                               default_policy=DRACTorchPolicy,
                               get_policy_class=get_policy_class,
                               make_policy_optimizer=choose_policy_optimizer,
                               validate_config=validate_config,
                               after_optimizer_step=update_kl,
                               after_train_result=warn_about_bad_reward_scales)
