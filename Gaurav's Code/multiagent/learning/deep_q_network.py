
import numpy as np
import pandas as pd
import tensorflow as tf

from multiagent.utilities.logging import getLogger
import multiagent.utilities.visualization as visualization
logger = getLogger(__name__)


class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        n_features,
        variable_identifier,
        learning_rate=0.01,
        gamma=0.9,
        epsilon=0.9,
        lambda_decay=0.001,
        replace_target_iter=100,
        memory_size=500,
        batch_size=64,
        epsilon_increment=None,
        epsilon_max=1.0,
        epsilon_min=0.01,
        output_visualization=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_decay = lambda_decay
        # self.epsilon_max = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_increment
        self.output_visualization = output_visualization

        # self.epsilon = 0 if epsilon_increment is not None else self.epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon
        self.learn_step_counter = 0
        self.memory_counter = 0

        self.variable_identifier = variable_identifier
        if(int(variable_identifier) < 0):
            logger.warning(
                "Variable_identifier %s likely invalid.", variable_identifier)

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()
        t_params = tf.get_collection(self._get_key('target_net_params'))
        e_params = tf.get_collection(self._get_key('eval_net_params'))
        self.replace_target_op = [tf.assign(t, e)
                                  for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        if self.output_visualization:
            # tf.summary.FileWriter(
            #    "logs/"+self.variable_identifier+"/", self.sess.graph)
            self.train_writer = tf.summary.FileWriter(
                visualization.train_path()+"/"+self.variable_identifier,
                self.sess.graph
            )
            # self.merge = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        self._compute_epsilon()

    def _compute_epsilon(self):
        self.epsilon = self.epsilon_min + \
            (self.epsilon_max - self.epsilon_min) * \
            np.exp(-self.lambda_decay * self.learn_step_counter)

    def _build_net(self):

        self.s = tf.placeholder(
            tf.float32, [None, self.n_features], name=self._get_key('state'))

        self.q_target = tf.placeholder(
            tf.float32, [None, self.n_actions], name=self._get_key('Q_target'))

        with tf.variable_scope(self._get_key('eval_net')):

            c_names, n_l1, w_initializer, b_initializer = \
                [self._get_key('eval_net_params'), tf.GraphKeys.GLOBAL_VARIABLES], 15, \
                tf.random_normal_initializer(
                    0., 0.3), tf.constant_initializer(0.1)  # config of layers

            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope(self._get_key('loss')):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_eval))
            if(self.output_visualization):
                self.loss_scalar = tf.summary.scalar("loss", self.loss)
        with tf.variable_scope(self._get_key('train')):
            self._train_op = tf.train.RMSPropOptimizer(
                self.learning_rate, momentum=0.9).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name=self._get_key('state_'))    # input
        with tf.variable_scope(self._get_key('target_net')):
            # c_names(collections_names) are the collections to store variables
            c_names = [self._get_key(
                'target_net_params'), tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def _get_key(self, key):
        return key + '_' + self.variable_identifier

    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(
                self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # logger.debug('Target params replaced.')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + \
            self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        if(self.output_visualization):
            summary, _ = self.sess.run([self.loss_scalar, self._train_op],
                                       feed_dict={self.s: batch_memory[:, :self.n_features],
                                                  self.q_target: q_target})
            self.train_writer.add_summary(summary, self.learn_step_counter)
            self.train_writer.flush()
        else:
            _ = self.sess.run([self._train_op],
                              feed_dict={self.s: batch_memory[:, :self.n_features],
                                         self.q_target: q_target})

        # increasing epsilon
        # self.epsilon = self.epsilon + \
        #    self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        self._compute_epsilon()
