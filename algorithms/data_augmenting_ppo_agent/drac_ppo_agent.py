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


def drac_init(policy, observation_space, action_space, options):
    setup_mixins(policy, observation_space, action_space, options)

    # Set up drac variables.
    drac_options = options["model"]["custom_options"]["auto_drac_options"]
    policy.always_use_transforms = drac_options["always_use_transforms"]
    policy.choose_between_transforms = []
    for t in drac_options["choose_between_transforms"]:
        if isinstance(t, list):
            t = tuple(t)
        policy.choose_between_transforms.append(t)
    policy.mean_update_rate = drac_options["mean_update_rate"]
    policy.action_value_update_rate = drac_options["action_value_update_rate"]
    policy.action_value_mode = drac_options["action_value_mode"]
    policy.ucb_c = drac_options["mode_to_ucb_c"][policy.action_value_mode]
    policy.sample_action_every_num_updates = drac_options["sample_action_every_num_updates"]

    policy.action_counts = {t: 0 for t in policy.choose_between_transforms}
    policy.action_values = {
        t: collections.deque(maxlen=policy.sample_action_every_num_updates * 10)
        for t in policy.choose_between_transforms
    }
    # Having two values allows for a better estimate of the action values (based
    # on the mean reward prior to the current set of minibatches.
    policy.prev_mean_reward = 0
    policy.next_mean_reward = 0
    policy.prev_action = None
    # The number of minibatches / policy updates made so far.
    policy.num_minibatches = 0
    policy.ucb_timestep = 0


def drac_loss_fn(policy, model, dist_class, train_batch):
    policy.num_minibatches += 1

    # for prints:
    cur_mean_reward = None

    # Update Q(a) and any other statistics required to compute it.
    if policy.action_value_mode == "mean_reward":
        action_value = train_batch["rewards"].detach().cpu().numpy().mean()
    elif policy.action_value_mode == "sum_reward":
        action_value = train_batch["rewards"].detach().cpu().numpy().sum()
    elif policy.action_value_mode == "mean_reward_advantage":
        cur_mean_reward = None
        if policy.prev_action is not None:
            cur_mean_reward = train_batch["rewards"].detach().cpu().numpy().mean()
            policy.next_mean_reward = (policy.mean_update_rate * cur_mean_reward +
                                       (1 - policy.mean_update_rate) * policy.next_mean_reward)
            action_value = cur_mean_reward - policy.prev_mean_reward
    elif policy.action_value_mode == "bootstrapped_value":
        action_value = train_batch["vf_preds"].detach().cpu().numpy().mean()
    else:
        raise NotImplementedError(f"Action value mode not implemented: {policy.action_value_mode}")

    if policy.prev_action is not None:
        policy.action_values[policy.prev_action].append(action_value)
        # policy.action_values[policy.prev_action] = (
        #     policy.action_value_update_rate * action_value +
        #     (1 - policy.action_value_update_rate) * policy.action_values[policy.prev_action])

    # Decide whether this is the first minibatch in a given set of updates.
    # If so update the options with a newly-selected set of transforms.
    if (policy.num_minibatches - 1) % policy.sample_action_every_num_updates == 0:
        action_values = {}
        for (k, v) in policy.action_values.items():
            if len(v) > 0:
                action_values[k] = np.mean(v)
            else:
                action_values[k] = 0

        print(f"drac_loss_fn, policy.num_minibatches: {policy.num_minibatches}")
        print(f"policy.ucb_timestep: {policy.ucb_timestep}")
        print(f"cur_mean_reward: {cur_mean_reward}")
        print(f"policy.action_values: {action_values}")
        print(f"policy.prev_mean_reward: {policy.prev_mean_reward}")
        print(f"policy.next_mean_reward: {policy.next_mean_reward}")

        policy.ucb_timestep += 1

        # Select the UCB action.
        best_score = -np.inf
        best_transform = None
        for transform in policy.choose_between_transforms:
            score = (action_values[transform] + policy.ucb_c *
                     (np.log(policy.ucb_timestep) / (policy.action_counts[transform] + 1e-8))**0.5)
            print(f"transform: {transform}, score: {score}")
            if score > best_score:
                best_score = score
                best_transform = transform
        assert best_transform is not None
        policy.action_counts[best_transform] += 1
        policy.prev_action = best_transform

        print(f"policy.action_counts: {policy.action_counts}")

        # Update the transforms that are actually used in the options based on the selected action.
        if isinstance(best_transform, str):
            formatted_best_transform = [best_transform]
        elif isinstance(best_transform, tuple):
            formatted_best_transform = list(best_transform)
        else:
            raise ValueError(f"Invalid best transform: {best_transform}")

        model.data_augmentation_options["transforms"] = (policy.always_use_transforms +
                                                         formatted_best_transform)
        print(f"transforms: {model.data_augmentation_options['transforms']}")

    # Perform any periodic updates depending on the value mode.
    # This runs at the last minibatch.
    if policy.num_minibatches % policy.sample_action_every_num_updates == 0:
        print("Updating prev mean reward!")
        if policy.action_value_mode == "mean_reward":
            pass
        elif policy.action_value_mode == "sum_reward":
            pass
        elif policy.action_value_mode == "bootstrapped_value":
            pass
        elif policy.action_value_mode == "mean_reward_advantage":
            policy.prev_mean_reward = policy.next_mean_reward
        else:
            raise NotImplementedError(
                f"Action value mode not implemented: {policy.action_value_mode}")

    return data_augmenting_loss(policy, model, dist_class, train_batch)


def drac_stats_fn(policy, train_batch):
    stats = kl_and_loss_stats(policy, train_batch)
    drac_stats = {}
    drac_stats.update({f"drac_{k}_count": v for (k, v) in policy.action_counts.items()})
    drac_stats.update({f"drac_{k}_value": np.mean(v) for (k, v) in policy.action_values.items()})
    drac_stats["drac_mean_reward"] = policy.prev_mean_reward
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
