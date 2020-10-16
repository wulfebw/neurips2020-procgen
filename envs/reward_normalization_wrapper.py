"""Took all this from https://github.com/MadryLab/implementation-matters"""

import gym
import numpy as np


class RunningStat:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.s = 0

    def add(self, v):
        self.n += 1
        if self.n == 1:
            self.m = v
        else:
            old_m = self.m
            self.m = self.m + (v - self.m) / self.n
            self.s = self.s + (v - old_m) * (v - self.m)

    @property
    def mean(self):
        return self.m

    @property
    def var(self):
        return self.s / (self.n - 1) if self.n > 1 else self.m**2

    @property
    def std(self):
        return np.sqrt(self.var)

    def __repr__(self):
        return f"Running stat with count: {self.n}, mean: {self.mean:.4f}, variance: {self.var:.4f}"


class DiscountedReturnStdEstimator:
    def __init__(self, discount):
        self.discount = discount
        self.reset()

    def __call__(self, x):
        self.ret = self.ret * self.discount + x
        self.rs.add(self.ret)
        return self.rs.std

    def reset(self):
        self.rs = RunningStat()
        self.ret = 0


class RewardNormalizationWrapper(gym.Wrapper):
    def __init__(self, env, discount):
        super().__init__(env)
        self.estimator = DiscountedReturnStdEstimator(discount)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info["rew_norm_g"] = self.estimator(rew)
        return obs, rew, done, info

    def reset(self):
        self.estimator.reset()
        return self.env.reset()
