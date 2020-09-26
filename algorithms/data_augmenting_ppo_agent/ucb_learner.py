import numpy as np


class UCBLearner:
    def __init__(self,
                 action_labels,
                 num_steps_per_update,
                 mean_reward_alpha,
                 q_alpha,
                 lmbda,
                 ucb_c,
                 verbose=False):
        self.action_labels = action_labels
        self.num_actions = len(action_labels)
        self.num_steps_per_update = num_steps_per_update
        self.mean_reward_alpha = mean_reward_alpha
        self.q_alpha = q_alpha
        self.lmbda = lmbda
        self.ucb_c = ucb_c
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
        return (self.num_steps - 1) % self.num_steps_per_update == 0

    def compute_reward(self):
        assert len(self.cur_rewards) > 0
        cur_mean_reward = np.mean(self.cur_rewards)

        # Compute the reward for this action-timestep as something like the advantage.
        r = cur_mean_reward - self.overall_mean_reward

        # Update overall mean.
        self.overall_mean_reward = self.overall_mean_reward + self.mean_reward_alpha * (
            cur_mean_reward - self.overall_mean_reward)

        # Reset current reward buffer.
        self.cur_rewards = []

        return r, dict(current_reward=r, overall_mean_reward=self.overall_mean_reward)

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
