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
