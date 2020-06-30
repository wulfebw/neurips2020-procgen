import cv2
import gym
import numpy as np


class FrameDiff(gym.Wrapper):
    def __init__(self, env, grayscale=True, dt=1):
        super().__init__(env)
        self.grayscale = grayscale
        self.dt = dt

        self.dtype = np.int16

        w, h, c = env.observation_space.shape
        self.num_diff_channels = 1 if self.grayscale else c
        # Store the original observations as np.uint8 to use less memory.
        self.framebuff = np.zeros((w, h, self.num_diff_channels * (self.dt + 1)), dtype=np.uint8)

        self.high_value = 255

        low = np.zeros((w, h, c + self.num_diff_channels), dtype=self.dtype)
        low[:, :, :self.num_diff_channels] = 0
        high = np.ones((w, h, c + self.num_diff_channels), dtype=self.dtype) * self.high_value
        high[:, :, :self.num_diff_channels] = self.high_value

        self.diffedobs = np.zeros(low.shape, low.dtype)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.dtype)

    def _frame_difference(self):
        diff = (self.framebuff[:, :, -self.num_diff_channels:].astype(np.int16) -
                self.framebuff[:, :, :self.num_diff_channels].astype(np.int16))
        diff = (diff + self.high_value) // 2
        return diff

    def _insert_obs_into_frame_buffer(self, obs):
        self.framebuff = np.roll(self.framebuff, shift=-self.num_diff_channels, axis=-1)

        if self.grayscale:
            self.framebuff[:, :, -1] = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            self.framebuff[:, :, -self.num_diff_channels:] = obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self._insert_obs_into_frame_buffer(obs)
        self.diffedobs[:, :, :self.num_diff_channels] = self._frame_difference()
        self.diffedobs[:, :, -obs.shape[-1]:] = obs

        # self._plot(obs)

        return self.diffedobs, rew, done, info

    def reset(self):
        self.diffedobs[:] = 0
        self.framebuff[:] = 0
        obs = self.env.reset()

        # On reset, set every image in the buffer to the reset image.
        # This ensures the frame difference is always the difference between some frames at least.
        stored_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) if self.grayscale else obs
        for i in range(self.dt + 1):
            if self.grayscale:
                self.framebuff[:, :, self.num_diff_channels * i] = stored_obs
            else:
                start = self.num_diff_channels * i
                end = self.num_diff_channels * (i + 1)
                self.framebuff[:, :, start:end] = stored_obs

        self.diffedobs[:, :, :self.num_diff_channels] = self._frame_difference()
        self.diffedobs[:, :, -obs.shape[-1]:] = obs
        return self.diffedobs

    def _plot(self, obs):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 6, figsize=(20, 6))
        fig.suptitle("FrameDifference Plot")
        axs[0].imshow(np.squeeze(self.framebuff[:, :, :self.num_diff_channels]))
        axs[0].set_title("oldest frame")
        axs[1].imshow(np.squeeze(self.framebuff[:, :, -self.num_diff_channels:]))
        axs[1].set_title("newest frame")
        axs[2].imshow(np.squeeze(self.diffedobs[:, :, :self.num_diff_channels]))
        axs[2].set_title("frame diff")
        axs[3].imshow(obs)
        axs[3].set_title("newest obs")
        axs[4].hist(self.diffedobs[:, :, :self.num_diff_channels].flatten())
        axs[4].set_title("Frame difference histogram")
        axs[5].hist(self.diffedobs[:, :, self.num_diff_channels:].flatten())
        axs[5].set_title("Observation histogram")
        print(self.diffedobs[:, :, :self.num_diff_channels].min())
        print(self.diffedobs[:, :, :self.num_diff_channels].max())
        print()
        plt.show()
