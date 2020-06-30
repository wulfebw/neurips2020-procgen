from collections import deque

import cv2
import gym
import numpy as np
import scipy.ndimage


class FrameStackPhaseCorrelation(gym.Wrapper):
    def __init__(self, env, phase_correlate=False):
        """Stack k last frames."""
        gym.Wrapper.__init__(self, env)
        self.phase_correlate = phase_correlate

        self.frames = deque([], maxlen=3)
        self.grayscale_frames = deque([], maxlen=3)

        h, w, c = env.observation_space.shape
        # Observation space is two frames, two timesteps apart.
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(h, w, c * 2),
                                                dtype=env.observation_space.dtype)

        self.aligned = np.zeros((h, w, c), dtype=np.uint8)

    def reset(self):
        obs = self.env.reset()
        if self.phase_correlate:
            grayscale_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY).astype(np.float32)
        for _ in range(3):
            self.frames.append(obs)
            if self.phase_correlate:
                self.grayscale_frames.append(grayscale_obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        if self.phase_correlate:
            self.grayscale_frames.append(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY).astype(np.float32))
        return self._get_obs(), reward, done, info

    def _get_obs_frames(self):
        if self.phase_correlate:
            shift, conf = cv2.phaseCorrelate(self.grayscale_frames[0], self.grayscale_frames[2])
            if np.linalg.norm(shift) > 0.25:
                # print(f"shift: {shift}, confidence: {conf}")

                shift = tuple(reversed(shift)) + (0, )
                scipy.ndimage.shift(self.frames[0], shift=shift, output=self.aligned)

                # self._plot(self.aligned, self.frames[2])

                return (self.aligned, self.frames[2])

        return (self.frames[0], self.frames[2])

    def _get_obs(self):
        assert len(self.frames) == 3

        # self._plot(*self._get_obs_frames())

        return np.concatenate(self._get_obs_frames(), axis=2)

    def _plot(self, aligned, current):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(2 * 6, 1 * 6))
        axs[0].imshow(aligned)
        axs[1].imshow(current)
        plt.show()
