import collections
import logging

import numpy as np
import scipy.signal as signal

import ray
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo.ppo import (DEFAULT_CONFIG, validate_config, update_kl,
                                      warn_about_bad_reward_scales, choose_policy_optimizer)
from ray.rllib.agents.ppo.ppo_tf_policy import postprocess_ppo_gae, setup_config
from ray.rllib.agents.ppo.ppo_torch_policy import (kl_and_loss_stats, vf_preds_fetches,
                                                   setup_mixins, KLCoeffMixin, ValueNetworkMixin,
                                                   PPOLoss)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.torch_ops import sequence_mask

from algorithms.data_augmentation.data_augmentation import apply_data_augmentation
from algorithms.data_augmenting_ppo_agent.ppo_utils import (compute_running_mean_and_variance,
                                                            RunningStat,
                                                            ExpWeightedMovingAverageStat)
from algorithms.data_augmenting_ppo_agent.ucb_learner import UCBLearner
from algorithms.data_augmenting_ppo_agent.sync_phasic_optimizer import SyncPhasicOptimizer

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)


def compute_ppo_loss(policy, dist_class, model, train_batch, action_dist, state):
    mask = None
    if state:
        max_seq_len = torch.max(train_batch["seq_lens"])
        mask = sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = torch.reshape(mask, [-1])

    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
    )

    return policy.loss_obj.loss


def update_transforms(policy, model, train_batch):
    auto_drac_opt = policy.config["auto_drac_options"]
    if not auto_drac_opt["active"]:
        return

    transform_index, transform_info = policy.transform_selector.step(
        list(train_batch["rewards"].detach().cpu().numpy()))

    # If the index is None, then no transform can be selected yet.
    if transform_index is None:
        return

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


def drac_data_augmenting_loss(policy,
                              model,
                              dist_class,
                              train_batch,
                              drac_weight,
                              drac_value_weight=1,
                              drac_policy_weight=1,
                              recurrent_repeat_transform=True):
    assert len(model.data_augmentation_options["transforms"]) > 0

    # If learning to select between transforms, do that first.
    update_transforms(policy, model, train_batch)

    no_aug_logits, no_aug_state = model.from_batch(train_batch)
    no_aug_action_dist = dist_class(no_aug_logits, model)
    no_aug_action_dist_detached = dist_class(no_aug_logits.detach(), model)
    no_aug_value = model.value_function().detach()

    policy_loss = compute_ppo_loss(policy, dist_class, model, train_batch, no_aug_action_dist,
                                   no_aug_state)

    if policy.is_recurrent() and recurrent_repeat_transform:
        obs = train_batch["obs"]
        b_flat, h, w, c = obs.shape
        t = policy.max_seq_len
        obs = obs.reshape(-1, t, h, w, c)
        obs = obs.permute(0, 2, 3, 1, 4)
        obs = obs.reshape(-1, h, w, c * t)

        obs, policy_weight_mask = apply_data_augmentation(obs, model.data_augmentation_options)

        # Repeat the policy weight mask `t` times so it gets applied to each timestep.
        policy_weight_mask = policy_weight_mask.repeat_interleave(t)

        debugging = False
        if debugging:
            import matplotlib.pyplot as plt
            num_samples_to_plot = 10
            for sample in range(num_samples_to_plot):
                fig, axs = plt.subplots(t, 1, figsize=(4, t * 3))
                for ti in range(t):
                    start = ti * 3
                    end = start + 3
                    axs[ti].imshow(obs[sample, :, :, start:end].detach().cpu().long().numpy())
                    axs[ti].set_title(f"sample {sample+1} / {num_samples_to_plot}, timestep: {ti}")
                fig.subplots_adjust(hspace=0.75)
                plt.show()
                plt.close()

        obs = obs.reshape(-1, h, w, t, c)
        obs = obs.permute(0, 3, 1, 2, 4)
        obs = obs.reshape(b_flat, h, w, c)
        train_batch["obs"] = obs
    else:
        train_batch["obs"], policy_weight_mask = apply_data_augmentation(
            train_batch["obs"], model.data_augmentation_options)

    model.norm_layers_active = True
    aug_logits, aug_state = model.from_batch(train_batch)
    model.norm_layers_active = False
    aug_action_dist = dist_class(aug_logits, model)
    aug_value = model.value_function()

    data_aug_value_loss = 0.5 * ((no_aug_value - aug_value)**2).mean()
    data_aug_policy_loss = (aug_action_dist.kl(no_aug_action_dist_detached) *
                            policy_weight_mask).mean()
    data_aug_loss = (drac_value_weight * data_aug_value_loss +
                     drac_policy_weight * data_aug_policy_loss)

    policy.loss_obj.data_aug_loss = data_aug_loss

    return policy_loss + drac_weight * data_aug_loss


def simple_data_augmenting_loss(policy, model, dist_class, train_batch):
    assert len(model.data_augmentation_options["transforms"]) > 0

    train_batch["obs"] = apply_data_augmentation(train_batch["obs"],
                                                 model.data_augmentation_options)
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    return compute_ppo_loss(policy, dist_class, model, train_batch, action_dist, state)


def no_data_augmenting_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    return compute_ppo_loss(policy, dist_class, model, train_batch, action_dist, state)


def phasic_aux_loss(policy, model, dist_class, train_batch, use_data_aug):
    assert "pre_aux_logits" in train_batch

    if use_data_aug:
        assert len(model.data_augmentation_options["transforms"]) > 0
        train_batch["obs"], _ = apply_data_augmentation(train_batch["obs"],
                                                        model.data_augmentation_options)

    # Forward through model.
    model.norm_layers_active = True
    logits, _ = model.from_batch(train_batch)
    model.norm_layers_active = False
    new_action_dist = dist_class(logits, model)
    value = model.value_function()

    # Behavioral cloning loss.
    old_action_dist = dist_class(train_batch["pre_aux_logits"], model)
    bc_loss = old_action_dist.kl(new_action_dist).mean()

    # Value function loss.
    value_targets = train_batch["value_targets"]
    value_loss = 0.5 * ((value - value_targets)**2).mean()

    # Overall.
    aux_loss = bc_loss + value_loss

    # Set these to report later in metrics.
    policy.aux_bc_loss = bc_loss
    policy.aux_value_loss = value_loss
    policy.aux_loss = aux_loss

    return aux_loss


def phasic_policy_loss(policy, model, dist_class, train_batch, policy_loss_mode, detach_value_head):
    if detach_value_head:
        model.detach_value_head()
    if policy_loss_mode == "simple":
        ret = no_data_augmenting_loss(policy, model, dist_class, train_batch)
    elif policy_loss_mode == "drac":
        drac_options = model.data_augmentation_options["mode_options"].get("drac", {})
        ret = drac_data_augmenting_loss(policy, model, dist_class, train_batch, **drac_options)
    else:
        raise ValueError(policy_loss_mode)
    if detach_value_head:
        model.attach_value_head()
    return ret


def get_phase_from_train_batch(train_batch):
    assert "phase" in train_batch, "Train batch must contain phase key."
    if train_batch["phase"][0] == 0:
        return "policy"
    else:
        return "aux"


def phasic_data_augmenting_loss(policy, model, dist_class, train_batch, use_data_aug,
                                policy_loss_mode, detach_value_head):
    phase = get_phase_from_train_batch(train_batch)
    if phase == "policy":
        return phasic_policy_loss(policy, model, dist_class, train_batch, policy_loss_mode,
                                  detach_value_head)
    elif phase == "aux":
        return phasic_aux_loss(policy, model, dist_class, train_batch, use_data_aug)
    else:
        raise ValueError(phase)


def data_augmenting_loss(policy, model, dist_class, train_batch):
    mode = model.data_augmentation_options["mode"]
    mode_options = model.data_augmentation_options["mode_options"].get(mode, {})

    if mode == "none":
        return no_data_augmenting_loss(policy, model, dist_class, train_batch, **mode_options)
    elif mode == "simple":
        return simple_data_augmenting_loss(policy, model, dist_class, train_batch, **mode_options)
    elif mode == "drac":
        return drac_data_augmenting_loss(policy, model, dist_class, train_batch, **mode_options)
    elif mode == "phasic":
        return phasic_data_augmenting_loss(policy, model, dist_class, train_batch, **mode_options)
    else:
        raise ValueError(f"Invalid data augmenting mode: {model.data_augmentation_options['mode']}")


def suppress_nan_and_inf(d, replacement=0):
    for k, v in d.items():
        if torch.is_tensor(v):
            if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                d[k] = replacement
                print("Suppressing inf or nan stat!")
        elif np.any(np.isnan(v)) or np.any(np.isinf(v)):
            d[k] = replacement
            print("Suppressing inf or nan stat!")
    return d


def add_auto_drac_stats(policy, stats):
    auto_drac_opt = policy.config["auto_drac_options"]
    if not auto_drac_opt["active"] or not hasattr(policy, "transform_info"):
        return stats

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
    if "overall_mean_reward" in policy.transform_info:
        drac_stats["overall_mean_reward"] = policy.transform_info["overall_mean_reward"]
    stats.update(drac_stats)
    return stats


def add_phasic_stats(policy, stats):
    phasic_stats_keys = ["aux_bc_loss", "aux_value_loss", "aux_loss"]
    for key in phasic_stats_keys:
        if hasattr(policy, key):
            stats[key] = getattr(policy, key)
    return stats


def add_reward_normalization_stats(policy, train_batch, stats):
    stats["reward_norm_value_targets"] = train_batch["value_targets"].detach().mean().cpu().numpy()
    return stats


def data_augmenting_stats(policy, train_batch):
    stats = kl_and_loss_stats(policy, train_batch)
    if hasattr(policy.loss_obj, "data_aug_loss"):
        stats["drac_loss_unweighted"] = policy.loss_obj.data_aug_loss

    stats = add_reward_normalization_stats(policy, train_batch, stats)
    stats = add_auto_drac_stats(policy, stats)
    stats = add_phasic_stats(policy, stats)
    stats = suppress_nan_and_inf(stats)
    return stats


def apply_noop_penalty(sample_batch, options):
    """Detects and penalizes noop transitions.

    Accomplishes this in a simple manner. It checks for identical, consecutive
    observations, and if they exist then it applies a penalty.
    """
    assert "reward" in options
    reward_value = options["reward"]
    cur_obs = sample_batch[SampleBatch.CUR_OBS]
    next_obs = sample_batch[SampleBatch.NEXT_OBS]
    dones = sample_batch[SampleBatch.DONES]

    diffs = (cur_obs[..., -3:] - next_obs[..., -3:]).sum(axis=(1, 2, 3))
    noop_timesteps = diffs == 0
    reward = noop_timesteps * reward_value
    # Don't apply the reward on a terminal timestep.
    reward = reward * (1 - dones)
    sample_batch[SampleBatch.REWARDS] += reward

    debugging = False
    if debugging:
        print(reward)
        print("len obs ", len(cur_obs))
        actions = sample_batch[SampleBatch.ACTIONS]
        print("actions ", actions)
        print("dones ", dones)
        filepath = "/home/wulfebw/Desktop/scratch/obs.png"
        import matplotlib.pyplot as plt
        num_timesteps = min(2, len(actions))
        fig, axs = plt.subplots(num_timesteps * 2, 2, figsize=(12, 8))
        # for i, t in enumerate(range(len(cur_obs) - 1, len(cur_obs) - num_timesteps, -1)):
        for i, t in enumerate(range(num_timesteps)):
            index = i * 2
            action = actions[t]
            axs[index][0].set_title(f"t: {t} action: {action} cur :3")
            axs[index][0].imshow(cur_obs[t, :, :, :3])
            axs[index][1].set_title(f"t: {t} action: {action}  cur -3:")
            axs[index][1].imshow(cur_obs[t, :, :, -3:])
            axs[index + 1][0].set_title(f"t: {t} action: {action}  next :3")
            axs[index + 1][0].imshow(next_obs[t, :, :, :3])
            axs[index + 1][1].set_title(f"t: {t} action: {action}  next -3:")
            axs[index + 1][1].imshow(next_obs[t, :, :, -3:])
        plt.tight_layout
        plt.savefig(filepath)
        plt.close()
        import ipdb
        ipdb.set_trace()

    return sample_batch


def apply_state_revisitation_penalty(sample_batch, options):
    infos = sample_batch[SampleBatch.INFOS]
    if len(infos) == 0 or "occupancy_count" not in infos[0]:
        return sample_batch

    reward_value = options["reward"]
    counts = [info["occupancy_count"] for info in infos]
    dones = sample_batch[SampleBatch.DONES]
    rewards = (np.array(counts) - 1) * reward_value
    # Don't apply the reward on a terminal timestep.
    rewards = rewards * (1 - dones)
    sample_batch[SampleBatch.REWARDS] += rewards

    debugging = False
    if debugging:
        print("\ncounts: ", counts)
        print("intrinsic rewards: ", rewards)
        print("overall rewards: ", sample_batch[SampleBatch.REWARDS])

        should_plot = False
        if sum(rewards) != 0 and should_plot:
            import matplotlib.pyplot as plt
            cur_obs = sample_batch[SampleBatch.CUR_OBS]
            num_timesteps = len(counts)
            for t in range(num_timesteps):
                print("count: ", counts[t])
                print("reward: ", rewards[t])
                plt.imshow(cur_obs[t, :, :, -3:])
                plt.show()
                plt.close()
        elif sum(rewards) != 0 and not should_plot:
            import ipdb
            ipdb.set_trace()

    return sample_batch


def intrinsic_reward_postprocess_sample_batch(policy,
                                              sample_batch,
                                              other_agent_batches=None,
                                              episode=None):
    opt = policy.model.intrinsic_reward_options
    if opt.get("use_noop_penalty", False):
        assert "noop_penalty_options" in opt
        sample_batch = apply_noop_penalty(sample_batch, opt["noop_penalty_options"])
    if opt.get("use_state_revisitation_penalty", False):
        assert "state_revisitation_penalty_options" in opt
        sample_batch = apply_state_revisitation_penalty(sample_batch,
                                                        opt["state_revisitation_penalty_options"])
    return sample_batch


def normalize_rewards_openai(rewards, discount, eps=1e-8):
    """Normalizes based on something like the discounted return, but honestly no idea.

    It's from this paper: https://arxiv.org/pdf/2005.12729.pdf, which says it got it
    from the openai baselines implementation, but I have no idea why you would do this.
    """
    a = [1, -discount]
    b = [1]
    discounted_return_so_far = signal.lfilter(b, a, x=rewards)
    _, variances = compute_running_mean_and_variance(discounted_return_so_far)
    return rewards / (np.sqrt(variances) + eps)


def normalize_rewards_mean_std(rewards, eps=1e-8):
    return (rewards - np.mean(rewards)) / (np.std(rewards) + eps)


def normalize_rewards_running_mean_std(policy, rewards, eps=1e-8):
    assert hasattr(policy, "reward_norm_stats")
    policy.reward_norm_stats.add_all(rewards)
    return (rewards - policy.reward_norm_stats.mean) / (policy.reward_norm_stats.std + eps)


def normalize_rewards_running_return(policy, rewards, eps=1e-8):
    assert hasattr(policy, "reward_norm_stats")
    policy.reward_norm_stats.add(np.sum(rewards))
    return rewards / (policy.reward_norm_stats.std + eps)


def normalize_rewards_env_return(rewards, infos, eps=1e-8):
    return_stds = np.array([info["rew_norm_g"] for info in infos])
    return rewards / (return_stds + eps)


def reward_normalize_postprocess_sample_batch(policy,
                                              sample_batch,
                                              other_agent_batches=None,
                                              episode=None):

    opt = policy.config.get("reward_normalization_options", {"mode": "none"})
    if opt["mode"] == "none":
        pass
    elif opt["mode"] == "openai":
        sample_batch[SampleBatch.REWARDS] = normalize_rewards_openai(
            sample_batch[SampleBatch.REWARDS], policy.config["gamma"])
    elif opt["mode"] == "mean_std":
        sample_batch[SampleBatch.REWARDS] = normalize_rewards_mean_std(
            sample_batch[SampleBatch.REWARDS])
    elif opt["mode"] == "running_mean_std":
        sample_batch[SampleBatch.REWARDS] = normalize_rewards_running_mean_std(
            policy, sample_batch[SampleBatch.REWARDS])
    elif opt["mode"] == "running_return":
        sample_batch[SampleBatch.REWARDS] = normalize_rewards_running_return(
            policy, sample_batch[SampleBatch.REWARDS])
    elif opt["mode"] == "env_rew_norm":
        sample_batch[SampleBatch.REWARDS] = normalize_rewards_env_return(
            sample_batch[SampleBatch.REWARDS], sample_batch[SampleBatch.INFOS])
    else:
        raise NotImplementedError(f"Reward normalization mode not implemented: {opt['mode']}")
    return sample_batch


def postprocess_sample_batch(policy, sample_batch, other_agent_batches=None, episode=None):
    # In theory you might want to apply to intrinsic rewards _after_ normalizing so that you can
    # use the same reward values across environments. In practice, using the same value across
    # envs works well, and doing it beforehand means you don't have to change the intrinsic
    # reward value based on whether you're normalizing the rewards.
    sample_batch = intrinsic_reward_postprocess_sample_batch(
        policy, sample_batch, other_agent_batches=other_agent_batches, episode=episode)
    sample_batch = reward_normalize_postprocess_sample_batch(
        policy, sample_batch, other_agent_batches=other_agent_batches, episode=episode)
    return postprocess_ppo_gae(policy, sample_batch, other_agent_batches, episode)


def compute_global_grad_norm(param_groups, norm_type=2, device="cpu"):
    norms = []
    for param_group in param_groups:
        for param in param_group["params"]:
            if param.grad is not None:
                norms += [torch.norm(param.grad.detach(), norm_type)]
    return torch.norm(torch.stack(norms), norm_type).to(device)


def apply_grad_clipping_elementwise(policy, optimizer, loss):
    info = {}
    if policy.config.get("grad_clip_elementwise", None) is not None:
        info["before_ele_clip_global_grad_norm"] = compute_global_grad_norm(optimizer.param_groups)
        for param_group in optimizer.param_groups:
            nn.utils.clip_grad_value_(param_group["params"], policy.config["grad_clip_elementwise"])
        info["after_ele_clip_global_grad_norm"] = compute_global_grad_norm(optimizer.param_groups)
    return info


def my_apply_grad_clipping(policy, optimizer, loss):
    # Apply the gradient clipping elementwise first to prevent the larger gradients at the
    # end of the network from dominating after clipping the gradients by the global norm.
    info = apply_grad_clipping_elementwise(policy, optimizer, loss)

    # Update the grad clip value depending on the mode.
    if policy.config["grad_clip_options"]["mode"] == "adaptive":
        assert policy.config.get("grad_clip_elementwise", None) is not None
        if len(policy.prev_gradient_norms
               ) == policy.config["grad_clip_options"]["adaptive_buffer_size"]:
            # Compute the grad clip value as a percentile of the previous buffer_size gradient norms.
            grad_clip = np.percentile(policy.prev_gradient_norms,
                                      q=policy.config["grad_clip_options"]["adaptive_percentile"])
            # Clip the grad clip value to a reasonable range.
            grad_clip = np.clip(
                grad_clip,
                policy.config["grad_clip_options"]["adaptive_min"],
                policy.config["grad_clip_options"]["adaptive_max"],
            )
            # Update the grad clip value on the policy. This will take effect below.
            policy.config["grad_clip"] = grad_clip
            # Track the effective grad clip value as a metric.
            info["effective_grad_clip"] = grad_clip

        # Update buffer of gradients.
        current_gradient_norm = info["after_ele_clip_global_grad_norm"]
        policy.prev_gradient_norms.append(current_gradient_norm)

    # Apply gradient clipping per usual, possibly using the updated grad clip value.
    global_info = apply_grad_clipping(policy, optimizer, loss)
    if "grad_gnorm" in global_info:
        info["final_grad_global_norm"] = global_info["grad_gnorm"].to("cpu")

    return info


def after_init_fn(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)

    # Reward normalization.
    rew_norm_opt = policy.config["reward_normalization_options"]
    if rew_norm_opt["mode"] == "running_mean_std":
        policy.reward_norm_stats = RunningStat(max_count=1000)
    elif rew_norm_opt["mode"] == "running_return":
        policy.reward_norm_stats = ExpWeightedMovingAverageStat(alpha=rew_norm_opt["alpha"])
    elif rew_norm_opt["mode"] == "none":
        pass
    else:
        raise ValueError(f"Unsupported reward norm mode: {rew_norm_opt['mode']}")

    # Gradient clipping.
    grad_clip_opt = policy.config["grad_clip_options"]
    if grad_clip_opt["mode"] == "constant":
        pass
    elif grad_clip_opt["mode"] == "adaptive":
        policy.prev_gradient_norms = collections.deque(maxlen=grad_clip_opt["adaptive_buffer_size"])
    else:
        raise ValueError(f"Unsupported grad clip mode: {grad_clip_opt['mode']}")

    # Automatic data augmentation.
    auto_drac_opt = policy.config["auto_drac_options"]
    if auto_drac_opt["active"]:
        policy.always_use_transforms = auto_drac_opt["always_use_transforms"]
        policy.choose_between_transforms = []
        for t in auto_drac_opt["choose_between_transforms"]:
            if isinstance(t, list):
                t = tuple(t)
            policy.choose_between_transforms.append(t)
        if auto_drac_opt["learner_class"] == "ucb":
            policy.transform_selector = UCBLearner(
                policy.choose_between_transforms,
                **auto_drac_opt["ucb_options"],
                verbose=False,
            )
        else:
            raise NotImplementedError(f"Learner not implemented: {auto_drac_opt['learner_class']}")


# Custom params to be available in the policy.
# The code above assumes these fields exist in the config, so if you remove one make sure to
# reflect that change above. The reason for this is that I don't want to hardcode defaults in
# the code directly.
DEFAULT_CONFIG["grad_clip_elementwise"] = None
DEFAULT_CONFIG["grad_clip_options"] = {
    "mode": "constant",
    "adaptive_max": 1.0,
    "adaptive_min": 0.1,
    "adaptive_percentile": 95,
    "adaptive_buffer_size": 100,
}

DEFAULT_CONFIG["reward_normalization_options"] = {
    "mode": "none",
    "alpha": 0.01,
}

DEFAULT_CONFIG["auto_drac_options"] = {
    "active":
    False,
    "always_use_transforms": [],
    "choose_between_transforms": [
        "random_translate",
        "random_rotation",
        "random_flip_left_right",
        "random_flip_up_down",
        ["random_translate", "random_flip_left_right"],
    ],
    "learner_class":
    "ucb",
    "ucb_options": {
        # This is the number of minibatches per action selection.
        # It should match train_batch_size / sgd_minibatch_size * num_sgd_iter
        "num_steps_per_update": 64,
        "q_alpha": 0.01,
        "mean_reward_alpha": 0.1,
        # Eligibility trace weight. "1" would assign credit to every past action,
        # and "0" would assign credit to only the most recent action.
        "lmbda": 0.25,
        "ucb_c": 0.05,
        "internal_reward_mode": "return",
    },
}

DEFAULT_CONFIG["use_phasic_optimizer"] = True
DEFAULT_CONFIG["aux_loss_every_k"] = 8
DEFAULT_CONFIG["aux_loss_num_sgd_iter"] = 3
DEFAULT_CONFIG["aux_loss_start_after_num_steps"] = 0

DataAugmentingTorchPolicy = build_torch_policy(name="DataAugmentingTorchPolicy",
                                               get_default_config=lambda: DEFAULT_CONFIG,
                                               loss_fn=data_augmenting_loss,
                                               stats_fn=data_augmenting_stats,
                                               extra_action_out_fn=vf_preds_fetches,
                                               postprocess_fn=postprocess_sample_batch,
                                               extra_grad_process_fn=my_apply_grad_clipping,
                                               before_init=setup_config,
                                               after_init=after_init_fn,
                                               mixins=[
                                                   KLCoeffMixin,
                                                   ValueNetworkMixin,
                                                   EntropyCoeffSchedule,
                                                   LearningRateSchedule,
                                               ])


# Well, this is a bit of a hack, but oh well.
def get_optimizer(policy, config={"opt_type": "adam"}):
    lr = policy.config["lr"]
    if hasattr(policy.model, "optimizer_options"):
        config = policy.model.optimizer_options

    opt_type = config.get("opt_type", "adam")
    if opt_type == "adam":
        return torch.optim.Adam(policy.model.parameters(), lr=lr)
    if opt_type == "adamw":
        return torch.optim.Adam(policy.model.parameters(),
                                lr=lr,
                                weight_decay=config.get("weight_decay", 0),
                                amsgrad=config.get("amsgrad", False))
    elif opt_type == "sgd":
        return torch.optim.SGD(policy.model.parameters(),
                               lr=lr,
                               momentum=config.get("momentum", 0.9),
                               nesterov=True)
    else:
        raise ValueError(f"Invalid optimizer: {opt_type}")


DataAugmentingTorchPolicy.optimizer = get_optimizer


def get_policy_class(config):
    return DataAugmentingTorchPolicy


def my_validate_config(config):
    # I guess this is the best way to check for recurrent policy.
    if config["model"]["use_lstm"]:
        if config["rollout_fragment_length"] != config["model"]["max_seq_len"]:
            logger.warning(
                "When using a recurrent policy, you should use a `rollout_fragment_length` equal to the `max_seq_len`.\n"
                "The reason for this is that the minibatches are not shuffled between the `max_seq_len` subsequences.\n"
                "As a result a single minibatch will have heavily correlated samples.\n"
                "By setting `rollout_fragment_length` equal to the `max_seq_len` you avoid this problem because\n"
                "each env sequence only occurs once in the full set of batches.\n"
                "Here are the values provided:\n"
                f"rollout_fragment_length: {config['rollout_fragment_length']}\n"
                f"max_seq_len: {config['model']['max_seq_len']}\n")
    validate_config(config)


def phasic_choose_policy_optimizer(workers, config):
    if config["use_phasic_optimizer"]:
        return SyncPhasicOptimizer(
            workers,
            num_sgd_iter=config["num_sgd_iter"],
            train_batch_size=config["train_batch_size"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
            standardize_fields=["advantages"],
            aux_loss_every_k=config["aux_loss_every_k"],
            aux_loss_num_sgd_iter=config["aux_loss_num_sgd_iter"],
            aux_loss_start_after_num_steps=config["aux_loss_start_after_num_steps"],
        )
    else:
        return choose_policy_optimizer(workers, config)


DataAugmentingPPOTrainer = build_trainer(name="data_augmenting_ppo_trainer",
                                         default_config=DEFAULT_CONFIG,
                                         default_policy=DataAugmentingTorchPolicy,
                                         get_policy_class=get_policy_class,
                                         make_policy_optimizer=phasic_choose_policy_optimizer,
                                         validate_config=my_validate_config,
                                         after_optimizer_step=update_kl,
                                         after_train_result=warn_about_bad_reward_scales)
