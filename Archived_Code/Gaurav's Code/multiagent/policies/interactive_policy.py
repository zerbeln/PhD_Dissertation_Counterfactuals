from multiagent.policies.policy import Policy
import numpy as np
from pyglet.window import key


class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release
        self.done = 0

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]:
                u = 1
            if self.move[1]:
                u = 2
            if self.move[2]:
                u = 4
            if self.move[3]:
                u = 3
        elif self.env.discrete_action_space is False:
            u = np.zeros(2)
            u = np.array([0.1, 0.1])
            if np.random.randint(0, 10) > 8:
                self.done = not self.done
            if self.done:
                u = np.array([0.1, 0.1])
            else:
                u = np.array([-0.1, -0.1])
        else:
            u = np.zeros(5)  # 5-d because of no-move action
            sensitivity = 1 if self.done < 25 else 0  # 0.05
            if True in self.move:
                self.done += 1
            if self.move[0]:
                u[1] += 1.0 * sensitivity
            if self.move[1]:
                u[2] += 1.0 * sensitivity
            if self.move[3]:
                u[3] += 1.0 * sensitivity
            if self.move[2]:
                u[4] += 1.0 * sensitivity
            if True not in self.move:
                u[0] += 1.0 * sensitivity
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k == key.LEFT:
            self.move[0] = True
        if k == key.RIGHT:
            self.move[1] = True
        if k == key.UP:
            self.move[2] = True
        if k == key.DOWN:
            self.move[3] = True

    def key_release(self, k, mod):
        if k == key.LEFT:
            self.move[0] = False
        if k == key.RIGHT:
            self.move[1] = False
        if k == key.UP:
            self.move[2] = False
        if k == key.DOWN:
            self.move[3] = False
