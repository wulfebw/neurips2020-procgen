import numpy as np


def compute_running_mean_and_variance_really_naive(arr):
    means = []
    variances = []
    for i in range(len(arr)):
        mean = np.mean(arr[:i + 1])
        if i == 0:
            var = arr[i]**2
        else:
            var = np.var(arr[:i + 1], ddof=1)
        means.append(mean)
        variances.append(var)
    return means, variances


def compute_running_mean_and_variance_naive(arr):
    n = 0
    m = 0
    means = []
    s = 0
    variances = []
    for i, v in enumerate(arr):
        n += 1
        if i == 0:
            m = v
        else:
            prev_m = m
            m = m + (v - m) / n
            s = s + (v - prev_m) * (v - m)
        means.append(m)
        variances.append(s / (n - 1) if n > 1 else m**2)
    return means, variances


def compute_running_mean(arr):
    return np.cumsum(arr) / (np.arange(len(arr)) + 1) if len(arr) > 0 else []


def compute_running_mean_and_variance(arr):
    means = compute_running_mean(arr)
    s = 0
    variances = np.empty_like(arr)
    for i, v in enumerate(arr):
        if i == 0:
            variances[i] = v**2
        else:
            s = s + (v - means[i - 1]) * (v - means[i])
            variances[i] = s / i  # i is equal to n - 1
    return means, variances


def time_fn(f, runs, *args):
    import time
    s = time.time()
    for r in range(runs):
        f(*args)
    e = time.time()
    print(f"{e - s:.6f}")


class RunningStat:
    """Took this from the `Implementation Matters ...` paper.

    https://github.com/MadryLab/implementation-matters
    """
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

    def add_all(self, arr):
        for v in arr:
            self.add(v)

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


class ExpWeightedMovingAverageStat:
    """https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_variance_and_standard_deviation"""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.n = 0
        self.m = 0
        self.v = 0

    def add(self, x):
        self.n += 1

        if self.n == 1:
            self.m = x
        else:
            delta = x - self.m
            self.m += self.alpha * delta
            self.v = (1 - self.alpha) * (self.v + self.alpha * delta**2)

    @property
    def mean(self):
        return self.m

    @property
    def var(self):
        return self.v

    @property
    def std(self):
        return np.sqrt(self.var)

    def __repr__(self):
        return f"Exp weighted moving avg mean: {self.mean:.4f}, variance: {self.var:.4f}"


if __name__ == "__main__":
    np.random.seed(1)
    size = 64
    arr = np.random.rand(size)

    running_mean_really_naive, running_variance_really_naive = compute_running_mean_and_variance_really_naive(
        arr)
    running_mean_naive, running_variance_naive = compute_running_mean_and_variance_naive(arr)
    running_mean, running_variance = compute_running_mean_and_variance(arr)

    np.testing.assert_array_almost_equal(running_mean_naive, running_mean_really_naive)
    np.testing.assert_array_almost_equal(running_variance_naive, running_variance_really_naive)

    np.testing.assert_array_almost_equal(running_mean, running_mean_really_naive)
    np.testing.assert_array_almost_equal(running_variance, running_variance_really_naive)

    runs = 10000
    time_fn(compute_running_mean_and_variance_really_naive, runs, arr)
    time_fn(compute_running_mean_and_variance_naive, runs, arr)
    time_fn(compute_running_mean_and_variance, runs, arr)
