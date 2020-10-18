import numpy as np


class UCBLearner:
    def __init__(self,
                 action_labels,
                 num_steps_per_update,
                 mean_reward_alpha,
                 q_alpha,
                 lmbda,
                 ucb_c,
                 internal_reward_mode="return",
                 verbose=False):
        self.action_labels = action_labels
        self.num_actions = len(action_labels)
        self.num_steps_per_update = num_steps_per_update
        self.mean_reward_alpha = mean_reward_alpha
        self.q_alpha = q_alpha
        self.lmbda = lmbda
        self.ucb_c = ucb_c
        self.internal_reward_mode = internal_reward_mode
        self.verbose = verbose

        # Variables that are updated during execution.
        self.num_steps = 0
        self.ucb_timestep = 0
        self.cur_rewards = []
        self.overall_mean_reward = 0
        self.eligibility_traces = {ai: 0.0 for ai in range(self.num_actions)}
        self.q = {ai: 0.0 for ai in range(self.num_actions)}
        self.action_counts = {ai: 0 for ai in range(self.num_actions)}
        self.action = None
        self.info = None

    def step(self, rewards):
        """Steps the learner forward.

        Returns:
            The action (index) to take and info about the internals.
            It's possible either will be None, and if that's the case
            The caller should choose the action in some other manner
            (e.g., randomly).
        """
        self.num_steps += 1
        self.collect(rewards)
        if self.should_update():
            info = dict()
            update_info = self.update()
            info.update(update_info)
            self.action, action_info = self.select_action()
            info.update(action_info)
            self.report(info)
            self.info = info
        return self.action, self.info

    def collect(self, rewards):
        self.cur_rewards.extend(rewards)

    def should_update(self):
        """The goal is to perform an update once per overall training cycle.

        This should return true at the last batch of that training cycle.
        Why? Because that way we will have collected all the rewards from the training
        batch (actually we will have collected them multiple times potentially,
        but that's fine). Once we collect all the rewards, we want to perform an update,
        and select a new action to provide.

        `num_steps` is set to 1 on the first `step(...)` call (before this function is called).
        The next time we want to return True then is after the total number of minibatches.
        Say there's 4 of those = `self.num_steps_per_update`.
        Then this should return True is when `self.num_steps` = 4.
        """
        return self.num_steps % self.num_steps_per_update == 0

    def compute_internal_ucb_reward(self, cur_mean_reward):
        if self.internal_reward_mode == "advantage":
            # Compute the reward for this action-timestep as something like the advantage.
            r = cur_mean_reward - self.overall_mean_reward
            # Update overall mean.
            self.overall_mean_reward = self.overall_mean_reward + self.mean_reward_alpha * (
                cur_mean_reward - self.overall_mean_reward)
            info = dict(overall_mean_reward=self.overall_mean_reward)
        elif self.internal_reward_mode == "return":
            # Optimizing for "return" = discounted sum rewards is approximately the same
            # as optimizing for the undiscounted mean reward.
            r = cur_mean_reward
            info = {}
        return r, info

    def compute_reward(self):
        assert len(self.cur_rewards) > 0
        cur_mean_reward = np.mean(self.cur_rewards)
        # Reset current reward buffer.
        self.cur_rewards = []

        r, internal_reward_info = self.compute_internal_ucb_reward(cur_mean_reward)
        internal_reward_info.update(dict(current_internal_reward=r))
        return r, internal_reward_info

    def update(self):
        r, r_info = self.compute_reward()

        # Update the actions in accordance with their responsibilities
        for ai, et in self.eligibility_traces.items():
            self.q[ai] = self.q[ai] + self.q_alpha * et * (r - self.q[ai])

        info = dict(action_values=self.label_action_dict(self.q))
        info.update(r_info)
        return info

    def select_action(self):
        self.ucb_timestep += 1

        # Select the UCB action.
        best_score = -np.inf
        best_action = None
        action_scores = dict()
        for ai, q in self.q.items():
            exploration_bonus = self.ucb_c * np.sqrt(
                np.log(self.ucb_timestep) / (self.action_counts[ai] + 1e-8))
            score = q + exploration_bonus
            action_scores[ai] = score

            if score > best_score:
                best_score = score
                best_action = ai
        assert best_action is not None
        self.action_counts[best_action] += 1

        # Update the eligibility traces.
        for ai, et in self.eligibility_traces.items():
            self.eligibility_traces[ai] = et * self.lmbda + (1 if ai == best_action else 0)

        return best_action, dict(eligibility=self.label_action_dict(self.eligibility_traces),
                                 action_counts=self.label_action_dict(self.action_counts),
                                 action_scores=self.label_action_dict(action_scores))

    def label_action_dict(self, d):
        labeled_d = dict()
        for action, value in d.items():
            action_label = self.action_labels[action]
            labeled_d[action_label] = value
        return labeled_d

    def report(self, info):
        if not self.verbose:
            return
        print(f"UCB timestep: {self.ucb_timestep}")
        print(f"Current action: {self.action_labels[self.action]}")
        for key in ["action_values", "action_scores", "action_counts", "eligibility"]:
            print(key)
            for action, value in info[key].items():
                print(f"\t{str(action):<50}: {value:.6f}")
