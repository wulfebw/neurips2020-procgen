import gym
import numpy as np


class MeanNormalizeImageObs(gym.ObservationWrapper):
    def __init__(self, env, mean):
        super().__init__(env)
        self.mean = mean
        low = env.observation_space.low - self.mean
        high = env.observation_space.high - self.mean
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=low.dtype)

    def observation(self, obs):
        err_msg = f"Invalid shape, expected: {self.mean.shape[-1]}, actual: {obs.shape[-1]}"
        assert obs.shape[-1] == self.mean.shape[-1], err_msg
        # self._plot(obs)
        return obs - self.mean

    def _plot(self, obs):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle("MeanNormalizeImageObs Plot")
        axs[0, 0].imshow(obs[:, :, :3])
        axs[0, 0].set_title("unnormalized diff")
        axs[1, 0].hist(obs[:, :, :3].flatten())

        axs[0, 1].imshow(obs[:, :, 3:])
        axs[0, 1].set_title("unnormalized img")
        axs[1, 1].hist(obs[:, :, 3:].flatten())

        norm = obs - self.mean
        axs[0, 2].imshow(norm[:, :, :3])
        axs[0, 2].set_title("normalized diff")
        norm_diff = norm[:, :, :3]
        axs[1, 2].hist(norm_diff.flatten())
        axs[1, 2].set_title(f"min: {norm_diff.min()} max: {norm_diff.max()} std: {norm_diff.std():.2f}")

        axs[0, 3].imshow(norm[:, :, 3:])
        axs[0, 3].set_title("normalized img")
        norm_img = norm[:, :, 3:]
        axs[1, 3].hist(norm_img.flatten())
        axs[1, 3].set_title(f"min: {norm_img.min()} max: {norm_img.max()} std: {norm_img.std():.2f}")
        plt.show()
