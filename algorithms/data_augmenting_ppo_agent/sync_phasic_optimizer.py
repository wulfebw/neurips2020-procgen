import logging

import numpy as np

import ray
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.filter import RunningStat
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.sgd import do_minibatch_sgd
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.memory import ray_get_and_free

logger = logging.getLogger(__name__)

torch, _ = try_import_torch()


def delete_recurrent_keys(samples):
    """This is pretty hacky.

    Remove all the recurrent keys so we can shuffle the batch
    without breaking the downstream batch formatting.
    This works because in the extra_action_out_fn the required states info
    was already transferred to a different variable.
    """
    keys = list(samples.keys())
    for key in keys:
        for key_substring_to_delete in ["state_in", "state_out"]:
            if key_substring_to_delete in key:
                del samples.data[key]


class SyncPhasicOptimizer(PolicyOptimizer):
    def __init__(self,
                 workers,
                 num_sgd_iter=1,
                 train_batch_size=1,
                 sgd_minibatch_size=0,
                 standardize_fields=frozenset([]),
                 aux_loss_every_k=16,
                 aux_loss_num_sgd_iter=9,
                 aux_loss_start_after_num_steps=0,
                 aux_loss_sgd_minibatch_size=None):
        PolicyOptimizer.__init__(self, workers)

        self.update_weights_timer = TimerStat()
        self.standardize_fields = standardize_fields
        self.sample_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.throughput = RunningStat()
        self.num_sgd_iter = num_sgd_iter
        self.sgd_minibatch_size = sgd_minibatch_size
        self.train_batch_size = train_batch_size
        self.learner_stats = {}
        self.policies = dict(
            self.workers.local_worker().foreach_trainable_policy(lambda p, i: (i, p)))
        logger.debug("Policies to train: {}".format(self.policies))

        self.aux_loss_every_k = aux_loss_every_k
        self.aux_loss_num_sgd_iter = aux_loss_num_sgd_iter
        self.aux_loss_start_after_num_steps = aux_loss_start_after_num_steps
        self.aux_loss_sgd_minibatch_size = (aux_loss_sgd_minibatch_size
                                            if aux_loss_sgd_minibatch_size is not None else
                                            sgd_minibatch_size)

        self.memory = []
        # Assert that train batch size is divisible by sgd minibatch size to make populating
        # policy logits simpler.
        assert train_batch_size % sgd_minibatch_size == 0, (
            f"train_batch_size: {train_batch_size}"
            f"sgd_minibatch_size: {sgd_minibatch_size}")

    @override(PolicyOptimizer)
    def step(self):
        with self.update_weights_timer:
            if self.workers.remote_workers():
                weights = ray.put(self.workers.local_worker().get_weights())
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights)

        with self.sample_timer:
            samples = []
            while sum(s.count for s in samples) < self.train_batch_size:
                if self.workers.remote_workers():
                    samples.extend(
                        ray_get_and_free([e.sample.remote()
                                          for e in self.workers.remote_workers()]))
                else:
                    samples.append(self.workers.local_worker().sample())
            samples = SampleBatch.concat_samples(samples)
            self.sample_timer.push_units_processed(samples.count)

        # Unfortunate to have to hack it like this, but not sure how else to do it.
        # Setting the phase to zeros results in policy optimization, and to ones results in aux optimization.
        # These have to be added prior to the policy sgd.
        samples["phase"] = np.zeros(samples.count)

        # This class wants nothing to do with recurrent policies.
        delete_recurrent_keys(samples)

        with self.grad_timer:
            fetches = do_minibatch_sgd(samples, self.policies, self.workers.local_worker(),
                                       self.num_sgd_iter, self.sgd_minibatch_size,
                                       self.standardize_fields)
        self.grad_timer.push_units_processed(samples.count)

        if len(fetches) == 1 and DEFAULT_POLICY_ID in fetches:
            self.learner_stats = fetches[DEFAULT_POLICY_ID]
        else:
            self.learner_stats = fetches

        self.num_steps_sampled += samples.count
        self.num_steps_trained += samples.count

        if self.num_steps_sampled > self.aux_loss_start_after_num_steps:
            # Add samples to the memory to be provided to the aux loss.
            self._remove_unnecessary_data(samples)
            self.memory.append(samples)

            # Optionally run the aux optimization.
            if len(self.memory) >= self.aux_loss_every_k:
                samples = SampleBatch.concat_samples(self.memory)
                self._add_policy_logits(samples)
                # Ones indicate aux phase.
                samples["phase"] = np.ones_like(samples["phase"])
                do_minibatch_sgd(samples, self.policies, self.workers.local_worker(),
                                 self.aux_loss_num_sgd_iter, self.aux_loss_sgd_minibatch_size, [])
                self.memory = []

        return self.learner_stats

    def _remove_unnecessary_data(self,
                                 samples,
                                 keys_to_keep=set([
                                     SampleBatch.CUR_OBS,
                                     SampleBatch.PREV_ACTIONS,
                                     SampleBatch.PREV_REWARDS,
                                     "phase",
                                     Postprocessing.VALUE_TARGETS,
                                     "options",
                                 ])):
        for key in list(samples.keys()):
            if key not in keys_to_keep:
                del samples.data[key]

    def _add_policy_logits(self, samples):
        with torch.no_grad():
            policy = self.policies["default_policy"]

            all_logits = []
            for start in range(0, samples.count, self.sgd_minibatch_size):
                end = start + self.sgd_minibatch_size

                batch = samples.slice(start, end)
                batch["is_training"] = False
                batch = policy._lazy_tensor_dict(batch)

                logits, _ = policy.model.from_batch(batch)
                all_logits.append(logits.detach().cpu().numpy())

            samples["pre_aux_logits"] = np.concatenate(all_logits)

    @override(PolicyOptimizer)
    def stats(self):
        return dict(
            PolicyOptimizer.stats(self), **{
                "sample_time_ms": round(1000 * self.sample_timer.mean, 3),
                "grad_time_ms": round(1000 * self.grad_timer.mean, 3),
                "update_time_ms": round(1000 * self.update_weights_timer.mean, 3),
                "opt_peak_throughput": round(self.grad_timer.mean_throughput, 3),
                "sample_peak_throughput": round(self.sample_timer.mean_throughput, 3),
                "opt_samples": round(self.grad_timer.mean_units_processed, 3),
                "learner": self.learner_stats,
            })
