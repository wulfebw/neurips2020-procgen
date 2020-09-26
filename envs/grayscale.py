import cv2
import gym
import numpy as np


class Grayscale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        w, h, c = env.observation_space.shape
        observation_shape = (w, h, 1)
        low = np.zeros(observation_shape, dtype=np.uint8)
        high = np.ones(observation_shape, dtype=np.uint8) * 255
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint8)

    def observation(self, observation):
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)[:, :, None]

    def plot(self, observation, filepath):
        import matplotlib.pyplot as plt
        plt.imshow(observation[:, :, 0])
        plt.savefig(filepath)
        plt.close()


if __name__ == "__main__":
    import sys
    env = gym.make("procgen:procgen-plunder-v0")
    env = Grayscale(env)
    x = env.reset()
    env.plot(x, f"/home/wulfebw/Desktop/scratch/gray_test_0.png")
    for i in range(1, 500):
        sys.stdout.write(f"\r{i}")
        x, _, done, _ = env.step(env.action_space.sample())
        env.plot(x, f"/home/wulfebw/Desktop/scratch/gray_test_{i}.png")
