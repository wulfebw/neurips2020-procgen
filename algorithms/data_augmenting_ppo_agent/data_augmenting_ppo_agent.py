import collections

import numpy as np

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
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.torch_ops import sequence_mask

torch, nn = try_import_torch()

from algorithms.data_augmenting_ppo_agent.data_augmentation import (
    random_translate_via_index,
    random_cutout_color_fast,
    random_cutout,
    random_channel_drop,
    random_rotation,
    random_flip,
)

PRIORITY_ORDERED_TRANSFORMS = [
    "random_rotation",
    "random_flip_left_right",
    "random_flip_up_down",
    "random_cutout",
    "random_cutout_color",
    "random_channel_drop",
    "random_conv",
    "random_translate",
]


def sort_transforms(ts):
    return sorted(ts, key=lambda x: PRIORITY_ORDERED_TRANSFORMS.index(x))


def apply_transform(imgs, transform, options):
    """Transforms the provided images with the requested transform.

    Returns:
        A tuple of the transformed images and a mask of same length as images
        indicating whether to apply the policy loss on the given transformed images.
    """
    policy_weight_mask = torch.ones(len(imgs), dtype=torch.float32, device=imgs.device)
    if transform == "random_translate":
        imgs = random_translate_via_index(imgs, **options.get("random_translate_options", {}))
    elif transform == "random_cutout_color":
        imgs = random_cutout_color_fast(imgs, **options.get("random_cutout_color_options", {}))
    elif transform == "random_cutout":
        imgs = random_cutout(imgs, **options.get("random_cutout_options", {}))
    elif transform == "random_channel_drop":
        imgs = random_channel_drop(imgs, **options.get("random_channel_drop_options", {}))
    elif transform == "random_flip_up_down":
        imgs = random_flip(imgs, flip_axis=1, **options.get("random_flip_options", {}))
        policy_weight_mask = torch.zeros(len(imgs), dtype=torch.float32, device=imgs.device)
    elif transform == "random_flip_left_right":
        imgs = random_flip(imgs, flip_axis=2, **options.get("random_flip_options", {}))
        policy_weight_mask = torch.zeros(len(imgs), dtype=torch.float32, device=imgs.device)
    elif transform == "random_rotation":
        imgs = random_rotation(imgs, **options.get("random_rotation_options", {}))
        policy_weight_mask = torch.zeros(len(imgs), dtype=torch.float32, device=imgs.device)
    else:
        raise NotImplementedError(f"Transform not implemented {transform}")

    return imgs, policy_weight_mask


def apply_data_augmentation_independently(imgs, options):
    num_transforms = len(options["transforms"])
    assert num_transforms > 0
    num_samples = len(imgs)
    assert num_samples > num_transforms
    num_samples_per_transform = num_samples // num_transforms
    transform_indices = np.random.permutation(num_transforms)
    policy_weight_mask = torch.ones(len(imgs), dtype=torch.float32, device=imgs.device)

    for i, transform_index in enumerate(transform_indices):
        transform = options["transforms"][transform_index]
        start = i * num_samples_per_transform
        end = start + num_samples_per_transform
        if i == num_transforms - 1:
            end = num_samples
        imgs[start:end], policy_weight_mask[start:end] = apply_transform(
            imgs[start:end], transform, options)
    return imgs, policy_weight_mask


def apply_data_augmentation_stacked(imgs, options):
    raise NotImplementedError("need to implement policy weighting for some transforms")
    assert len(options["transforms"]) > 0
    for transform in sort_transforms(options["transforms"]):
        imgs = apply_transform(imgs, transform, options)
    return imgs


def apply_data_augmentation(imgs, options):
    aug_mode = options.get("augmentation_mode", "independent")
    if aug_mode == "independent":
        return apply_data_augmentation_independently(imgs, options)
    elif aug_mode == "stacked":
        return apply_data_augmentation_stacked(imgs, options)
    else:
        raise ValueError(f"Invalid augmentation mode: {aug_mode}")


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


def drac_data_augmenting_loss(policy,
                              model,
                              dist_class,
                              train_batch,
                              drac_weight,
                              drac_value_weight=1,
                              drac_policy_weight=1):
    assert len(model.data_augmentation_options["transforms"]) > 0

    no_aug_logits, no_aug_state = model.from_batch(train_batch)
    no_aug_action_dist = dist_class(no_aug_logits, model)
    no_aug_action_dist_detached = dist_class(no_aug_logits.detach(), model)
    no_aug_value = model.value_function().detach()

    policy_loss = compute_ppo_loss(policy, dist_class, model, train_batch, no_aug_action_dist,
                                   no_aug_state)

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


def data_augmenting_loss(policy, model, dist_class, train_batch):
    mode = model.data_augmentation_options["mode"]
    mode_options = model.data_augmentation_options["mode_options"].get(mode, {})

    if mode == "none":
        return no_data_augmenting_loss(policy, model, dist_class, train_batch, **mode_options)
    elif mode == "simple":
        return simple_data_augmenting_loss(policy, model, dist_class, train_batch, **mode_options)
    elif mode == "drac":
        return drac_data_augmenting_loss(policy, model, dist_class, train_batch, **mode_options)
    else:
        raise ValueError(f"Invalid data augmenting mode: {model.data_augmentation_options['mode']}")


def data_augmenting_stats(policy, train_batch):
    stats = kl_and_loss_stats(policy, train_batch)
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


def intrinsic_reward_postprocess_fn(policy, sample_batch, other_agent_batches=None, episode=None):
    opt = policy.model.intrinsic_reward_options
    if opt.get("use_noop_penalty", False):
        assert "noop_penalty_options" in opt
        sample_batch = apply_noop_penalty(sample_batch, opt["noop_penalty_options"])

    return postprocess_ppo_gae(policy, sample_batch, other_agent_batches, episode)


DataAugmentingTorchPolicy = build_torch_policy(name="DataAugmentingTorchPolicy",
                                               get_default_config=lambda: DEFAULT_CONFIG,
                                               loss_fn=data_augmenting_loss,
                                               stats_fn=data_augmenting_stats,
                                               extra_action_out_fn=vf_preds_fetches,
                                               postprocess_fn=intrinsic_reward_postprocess_fn,
                                               extra_grad_process_fn=apply_grad_clipping,
                                               before_init=setup_config,
                                               after_init=setup_mixins,
                                               mixins=[KLCoeffMixin, ValueNetworkMixin])


# Well, this is a bit of a hack, but oh well.
def get_optimizer(policy):
    config = policy.model.optimizer_options

    lr = policy.config["lr"]
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


DataAugmentingPPOTrainer = build_trainer(name="data_augmenting_ppo_trainer",
                                         default_config=DEFAULT_CONFIG,
                                         default_policy=DataAugmentingTorchPolicy,
                                         get_policy_class=get_policy_class,
                                         make_policy_optimizer=choose_policy_optimizer,
                                         validate_config=validate_config,
                                         after_optimizer_step=update_kl,
                                         after_train_result=warn_about_bad_reward_scales)
