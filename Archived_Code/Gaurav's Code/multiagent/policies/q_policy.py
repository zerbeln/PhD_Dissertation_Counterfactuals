from multiagent.policies.policy import Policy
import numpy as np
import random

from multiagent.learning.deep_q_network import DeepQNetwork


class QPolicy(Policy):
    def __init__(self, env, agent_index,
                 n_actions,
                 n_features,
                 learning_rate=0.001,
                 epsilon=0.9,
                 replace_target_iter=2000,
                 memory_size=3000,
                 epsilon_increment=None,  # 0.0002,
                 output_visualization=False):

        self.env = env
        self.brain = DeepQNetwork(
            n_actions=n_actions,
            n_features=n_features,
            variable_identifier=str(agent_index),
            learning_rate=learning_rate,
            epsilon=epsilon,
            replace_target_iter=replace_target_iter,
            memory_size=memory_size,
            epsilon_increment=epsilon_increment,
            output_visualization=output_visualization
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

        action_vector = action[:self.brain.n_actions]   # remove communication
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

    def _actions_4_(self, obs):

        discrete_action = self.brain.choose_action(obs)
        # actions in env are 0 to 4, 0 = no_action. Instead, just predict 0-3 and translate to 1-4 for env. Not using no_action
        discrete_action = discrete_action + 1
        # convert this action to a generic representation
        u = np.zeros(self.brain.n_actions + 1)
        u[discrete_action] += 1.0

        # create a vector with the communication chanel
        action_vector = np.concatenate([u, np.zeros(self.env.world.dim_c)])
        return action_vector

    def _transition_4_(self,
                       observation,
                       action,
                       reward,
                       next_observation,
                       done):

        # get discrete action from action vector

        # remove communication
        action_vector = action[:self.brain.n_actions + 1]  # +1 for no_action
        discrete_action = np.argmax(action_vector)
        # ignore no_action
        discrete_action = discrete_action - 1

        self.brain.store_transition(
            observation,
            discrete_action,
            reward,
            next_observation
        )
        self.transition_steps += 1

        # if(self.transition_steps % self.learn_in_steps == 0):
        self.brain.learn()
