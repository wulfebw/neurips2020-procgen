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


def episode_adversarial_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

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
    return model.custom_loss(policy.loss_obj.loss, train_batch[SampleBatch.EPS_ID])


def episode_adversarial_stats(policy, train_batch):
    stats = kl_and_loss_stats(policy, train_batch)
    stats.update(policy.model.metrics())
    return stats


EpisodeAdversarialTorchPolicy = build_torch_policy(name="EpisodeAdversarialPolicy",
                                                   get_default_config=lambda: DEFAULT_CONFIG,
                                                   loss_fn=episode_adversarial_loss,
                                                   stats_fn=episode_adversarial_stats,
                                                   extra_action_out_fn=vf_preds_fetches,
                                                   postprocess_fn=postprocess_ppo_gae,
                                                   extra_grad_process_fn=apply_grad_clipping,
                                                   before_init=setup_config,
                                                   after_init=setup_mixins,
                                                   mixins=[KLCoeffMixin, ValueNetworkMixin])


def get_policy_class(config):
    return EpisodeAdversarialTorchPolicy


EpisodeAdversarialTrainer = build_trainer(name="episode_adversarial",
                                          default_config=DEFAULT_CONFIG,
                                          default_policy=EpisodeAdversarialTorchPolicy,
                                          get_policy_class=get_policy_class,
                                          make_policy_optimizer=choose_policy_optimizer,
                                          validate_config=validate_config,
                                          after_optimizer_step=update_kl,
                                          after_train_result=warn_about_bad_reward_scales)
