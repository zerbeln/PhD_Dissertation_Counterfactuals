from multiagent.policies.policy import Policy
import numpy as np
import random

from multiagent.learning.pr_dqn import DQNPrioritizedReplay


class PRQPolicy(Policy):
    def __init__(self, env, agent_index,
                 n_actions,
                 n_features,
                 learning_rate=0.005,
                 epsilon=0.9,
                 replace_target_iter=500,
                 memory_size=10000,
                 epsilon_increment=None,  # 0.0002,
                 output_visualization=False):

        self.env = env
        self.brain = DQNPrioritizedReplay(
            n_actions=n_actions,
            n_features=n_features,
            variable_identifier=str(agent_index),
            learning_rate=learning_rate,
            e_greedy=epsilon,
            replace_target_iter=replace_target_iter,
            memory_size=memory_size,
            e_greedy_increment=epsilon_increment,
            batch_size=64
        )

        self.learn_in_steps = 10
        self.transition_steps = 0

    def action(self, obs):

        discrete_action = self.brain.choose_action(obs)

        # convert this action to a generic representation
        u = np.zeros(self.brain.n_actions)
        u[discrete_action] += 1.0

        # create a vector with the communication chanel
        action_vector = np.concatenate([u, np.zeros(self.env.world.dim_c)])
        return action_vector

    def transition(self,
                   observation,
                   action,
                   reward,
                   next_observation,
                   done):

        # get discrete action from action vector

        action_vector = action[:self.brain.n_actions]  # remove communication
        discrete_action = np.argmax(action_vector)

        self.brain.store_transition(
            observation,
            discrete_action,
            reward,
            next_observation
        )
        self.transition_steps += 1

        # if(self.transition_steps % self.learn_in_steps == 0):
        self.brain.learn()
