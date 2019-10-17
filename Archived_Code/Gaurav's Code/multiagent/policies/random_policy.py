from multiagent.policies.policy import Policy
import numpy as np
import random


class RandomPolicy(Policy):
    def __init__(self, env, agent_index):
        self.env = env

    def action(self, obs):
        # ignore observation and pick a random action.
        if self.env.discrete_action_input:
            u = random.choice([0, 1, 2, 3, 4])
        else:
            u = np.zeros(5)  # 5-d because of no-move action
            u[random.choice([0, 1, 2, 3, 4])] = 0.1
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])
