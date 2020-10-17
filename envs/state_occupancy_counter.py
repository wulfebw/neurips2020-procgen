import collections

import gym
import numpy as np


class StateOccupancyCounter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset()

    def reset(self):
        self.state_occupancy_counts = collections.Counter()
        obs = self.env.reset()
        self.update_state_occupancy_count(obs)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info["occupancy_count"] = self.update_state_occupancy_count(obs)
        return obs, rew, done, info

    def compute_obs_hash(self, obs):
        return hash(obs.tostring())

    def update_state_occupancy_count(self, obs):
        obs_hash = self.compute_obs_hash(obs)
        self.state_occupancy_counts[obs_hash] += 1
        return self.state_occupancy_counts[obs_hash]


if __name__ == "__main__":

    def time_function(f, num_iters, *args):
        import sys
        import time
        start = time.time()
        for itr in range(num_iters):
            sys.stdout.write(f"\r{itr + 1} / {num_iters}")
            f(*args)
        end = time.time()
        print(f"\nTook {end - start:0.4f} seconds")

    import copy

    num_imgs = 2048
    height = 64
    width = 64
    channels = 6
    num_iters = 100

    import gym
    env = gym.make("procgen:procgen-caveflyer-v0")
    env = StateOccupancyCounter(env)
    obs = []
    x = env.reset()
    obs.append(x)
    infos = []
    for i in range(1, num_imgs):
        x, _, done, info = env.step(env.action_space.sample())
        obs.append(x)
        infos.append(info)
    obs = np.array(obs)

    counts = [i["occupancy_count"] for i in infos]
    print(counts)
    print(collections.Counter(counts))
    import ipdb
    ipdb.set_trace()

    # def hash_obses(env, obs):
    #     for o in obs:
    #         env.compute_obs_hash(o)

    # time_function(hash_obses, 4000, env, obs)

    # imgs = result_a
    # import matplotlib.pyplot as plt
    # for ob, img in zip(obs, imgs):
    #     fig, axs = plt.subplots(2, 1, figsize=(16, 16))
    #     axs[0].imshow(ob[:, :, :3].detach().cpu().to(torch.uint8).numpy())
    #     axs[1].imshow(img[:, :, :3].detach().cpu().to(torch.uint8).numpy())
    #     plt.show()
