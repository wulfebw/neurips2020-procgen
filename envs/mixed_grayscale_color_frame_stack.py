import cv2
import gym
import numpy as np


class MixedGrayscaleColorFrameStack(gym.ObservationWrapper):
    """Stacks the latest frame as color and previous frames as grayscale."""
    def __init__(self, env, num_prev_frames=1):
        super().__init__(env)
        self.num_prev_frames = num_prev_frames

        w, h, c = env.observation_space.shape
        # This stores the frames in grayscale (we only need to store frames in grayscale not color).
        self.grayscale_frames = np.zeros((w, h, self.num_prev_frames), dtype=np.uint8)

        # Number of channels is (channels for latest in color) + (number of previous frames).
        observation_shape = (w, h, c + num_prev_frames)
        low = np.zeros(observation_shape, dtype=np.uint8)
        high = np.ones(observation_shape, dtype=np.uint8) * 255
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint8)

    def _insert_observation(self, observation):
        self.grayscale_frames = np.roll(self.grayscale_frames, shift=-1, axis=-1)
        self.grayscale_frames[:, :, -1] = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

    def _format_obseration(self, observation):
        return np.concatenate((self.grayscale_frames, observation), axis=-1)

    def observation(self, observation):
        observation_to_return = self._format_obseration(observation)
        self._insert_observation(observation)
        return observation_to_return

    def reset(self):
        self.grayscale_frames.fill(0)
        return super().reset()

    def plot(self, obs, filepath):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, self.num_prev_frames + 1, figsize=(12, 6))
        axs[0].imshow(obs[:, :, -3:])
        for i in range(1, self.num_prev_frames + 1):
            axs[i].imshow(obs[:, :, i - 1])
        plt.savefig(filepath)
        plt.close()


if __name__ == "__main__":
    import sys
    env = gym.make("procgen:procgen-plunder-v0")
    env = MixedGrayscaleColorFrameStack(env, num_prev_frames=2)
    x = env.reset()
    env.plot(x, f"/home/wulfebw/Desktop/scratch/test_0.png")
    for i in range(1, 500):
        sys.stdout.write(f"\r{i}")
        x, _, done, _ = env.step(env.action_space.sample())
        env.plot(x, f"/home/wulfebw/Desktop/scratch/test_{i}.png")
