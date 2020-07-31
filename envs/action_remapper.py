import gym


class ActionRemapper(gym.ActionWrapper):
    def __init__(self, env, action_indices):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._action_map = {
            new_index: orig_index
            for (new_index, orig_index) in enumerate(action_indices)
        }
        self.action_space = gym.spaces.Discrete(len(action_indices))

    def action(self, action):
        assert self.action_space.contains(action)
        return self._action_map[action]
