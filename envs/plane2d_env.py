import sys

import gym
# import matplotlib.pyplot as plt
import numpy as np


class Plane2d(gym.Env):
    """A simple 2D plane.

    Observation space:
        - continuous (x, y) position in plane.
    
    Action space:
        - discrete left/right/up/down

    Transition model:
        - Deterministic movement by some distance

    Reward model:
        - No rewards

    Horizon:
        - Finite horizon

    Initial state distribution:
        - Always start at (0, 0)
    """
    def __init__(self, horizon=100):
        low = np.array([-horizon, -horizon], dtype=np.float32)
        high = np.array([horizon, horizon], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.action_index_to_delta = {0: [1, 0], 1: [-1, 0], 2: [0, 1], 3: [0, -1]}
        self.horizon = horizon
        self.state = None
        self.t = None

    def reset(self):
        self.state = np.array([0, 0], np.float32)
        self.t = 0
        return self.state

    def step(self, action):
        assert self.action_space.contains(action)
        assert self.state is not None
        assert self.t is not None
        delta = self.action_index_to_delta[action]
        self.state += delta
        self.t += 1
        done = self.t >= self.horizon
        return self.state, 0, done, {}

    def render(self, return_rgb_array=False):
        if self.state is None or self.t is None:
            return

        plane = np.zeros((self.horizon * 2, self.horizon * 2))
        x, y = self.state.astype(np.int)
        x += self.horizon
        y += self.horizon
        plane[x, y] = 1

        if return_rgb_array:
            return plane
        else:
            plt.imshow(plane)
            plt.show()
            plt.close()


if __name__ == "__main__":
    env = Plane2d()
    env.reset()
    img = np.zeros((env.horizon * 2, env.horizon * 2))
    max_steps = 1000000
    for i in range(max_steps):
        sys.stdout.write(f"\r{i+1} / {max_steps}")
        obs, rew, done, info = env.step(env.action_space.sample())
        img += env.render(return_rgb_array=True)
        if done:
            env.reset()
    img /= np.sum(img)
    plt.imshow(img)
    plt.show()
