import collections
import os
import pathlib
import sys
import unittest

import numpy as np

# There's no package set up so do something hacky to import.
sys.path.append(str(pathlib.Path(__file__).parent.absolute().parent))
from ucb_learner import UCBLearner


class TestUCBLearner(unittest.TestCase):
    def test_converges(self):
        learner = UCBLearner(["a", "b", "c"],
                             num_steps_per_update=5,
                             mean_reward_alpha=0.1,
                             q_alpha=0.01,
                             lmbda=0.25,
                             ucb_c=0.005,
                             verbose=True)

        num_train_itr = 1000
        num_batch_itr = 5
        action = 0
        means = np.array([0.001, -0.001, 0.0005])
        sigmas = np.array([0.01, 0.02, 0.003])
        num_samples_per_action = 10
        num_samples_per_batch = 200
        rewards = collections.deque(maxlen=num_samples_per_batch)
        for train_itr in range(num_train_itr):
            for batch_itr in range(num_batch_itr):
                current_rewards = np.ones(num_samples_per_action) * means[action]
                current_rewards += train_itr / num_train_itr
                current_rewards += np.random.randn(num_samples_per_action) * sigmas[action]
                action, info = learner.step(list(current_rewards))


if __name__ == "__main__":
    unittest.main()
