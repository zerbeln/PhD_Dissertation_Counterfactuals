import numpy as np
import math
import sys
from parameters import parameters as p
import warnings
warnings.filterwarnings('ignore')


class Poi:
    def __init__(self, px, py, pval, pc, pq, p_id):
        self.poi_id = p_id
        self.x_position = px
        self.y_position = py
        self.value = pval
        self.coupling = pc
        self.quadrant = pq
        self.observer_distances = np.zeros(p["n_rovers"])


class Rover:
    def __init__(self, rov_id, rov_x, rov_y, rov_theta):
        # Rover Parameters -----------------------------------------------------------------------------------
        self.self_id = rov_id  # Rover's unique identifier
        self.x_pos = rov_x
        self.y_pos = rov_y
        self.theta_pos = rov_theta
        self.initial_pos = np.zeros(3)  # Keeps initial position of rover stored for quick reset
        self.dmax = p["dmax"]  # Maximum distance a rover can move each time step

        # Rover Sensor Characteristics -----------------------------------------------------------------------
        self.sensor_type = p["sensor_model"]  # Type of sesnors rover is equipped with
        self.sensor_range = p["observation_radius"]  # Distances which sensors can observe POI
        self.sensor_res = p["angle_res"]  # Angular resolution of the sensors
        self.delta_min = p["min_distance"]  # Lower bound for distance in sensor/utility calculations

        # Rover Data -----------------------------------------------------------------------------------------
        self.sensor_readings = np.zeros(p["n_inputs"], dtype=np.float128)  # Number of sensor inputs for Neural Network
        self.poi_distances = np.ones(p["n_poi"]) * 1000.00  # Records distances measured from sensors
        self.rover_actions = np.zeros(p["n_outputs"], dtype=np.float128)  # Motor actions from neural network outputs

        # Rover Motor Controller -----------------------------------------------------------------------------
        self.n_inputs = p["n_inputs"]
        self.n_outputs = p["n_outputs"]
        self.n_hnodes = p["n_hidden"]  # Number of nodes in hidden layer
        self.weights = {}
        self.input_layer = np.reshape(np.mat(np.zeros(self.n_inputs, dtype=np.float128)), [self.n_inputs, 1])
        self.hidden_layer = np.reshape(np.mat(np.zeros(self.n_hnodes, dtype=np.float128)), [self.n_hnodes, 1])
        self.output_layer = np.reshape(np.mat(np.zeros(self.n_outputs, dtype=np.float128)), [self.n_outputs, 1])

    def reset_rover(self):
        """
        Resets the rover to its initial position in the world
        """
        self.x_pos = self.initial_pos[0]
        self.y_pos = self.initial_pos[1]
        self.theta_pos = self.initial_pos[2]
        self.sensor_readings = np.zeros(self.n_inputs, dtype=np.float128)
        self.poi_distances = np.ones(p["n_poi"]) * 1000.00

    def step(self, world_x, world_y):
        """
        Rover executes current actions provided by neuro-controller (not using policy playbook)
        """
        # Get outputs from neuro-controller
        self.get_nn_outputs()

        # Update rover positions based on outputs and assign to dummy variables
        dx = 2 * self.dmax * (self.rover_actions[0] - 0.5)
        dy = 2 * self.dmax * (self.rover_actions[1] - 0.5)

        # Update X Position
        x = dx + self.x_pos
        if x < 0:
            x = 0
        elif x > world_x-1:
            x = world_x-1

        # Update Y Position
        y = dy + self.y_pos
        if y < 0:
            y = 0
        elif y > world_y-1:
            y = world_y-1

        self.x_pos = x
        self.y_pos = y

    def suggestion_step(self, world_x, world_y):
        """
        Rover executes current actions provided by neuro-controller (for suggestions training only)
        """

        # Update rover positions based on outputs and assign to dummy variables
        dx = 2 * self.dmax * (self.rover_actions[0] - 0.5)
        dy = 2 * self.dmax * (self.rover_actions[1] - 0.5)

        # Update X Position
        x = dx + self.x_pos
        if x < 0:
            x = 0
        elif x > world_x - 1:
            x = world_x - 1

        # Update Y Position
        y = dy + self.y_pos
        if y < 0:
            y = 0
        elif y > world_y - 1:
            y = world_y - 1

        self.x_pos = x
        self.y_pos = y

    def scan_environment(self, rovers, pois):
        """
        Constructs the state information that gets passed to the rover's neuro-controller
        """
        poi_state = self.poi_scan(pois)
        rover_state = self.rover_scan(rovers)

        for i in range(4):
            self.sensor_readings[i] = poi_state[i]
            self.input_layer[i, 0] = poi_state[i]
            self.sensor_readings[4 + i] = rover_state[i]
            self.input_layer[4 + i, 0] = rover_state[i]

    def poi_scan(self, pois):
        """
        Rover queries scanner that detects POIs
        """
        poi_state = np.zeros(int(360.0 / self.sensor_res))
        temp_poi_dist_list = [[] for _ in range(int(360.0 / self.sensor_res))]

        # Log POI distances into brackets
        poi_id = 0
        for pk in pois:
            angle, dist = self.get_angle_dist(self.x_pos, self.y_pos, pois[pk].x_position, pois[pk].y_position)

            self.poi_distances[poi_id] = math.sqrt(dist)  # Record distance for sensor information
            bracket = int(angle / self.sensor_res)
            if bracket > 3:
                bracket -= 4
            temp_poi_dist_list[bracket].append(pois[pk].value / dist)
            poi_id += 1

        # Encode POI information into the state vector
        for bracket in range(int(360 / self.sensor_res)):
            num_poi_bracket = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
            if num_poi_bracket > 0:
                if self.sensor_type == 'density':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi_bracket  # Density Sensor
                elif self.sensor_type == 'summed':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket])  # Summed Distance Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                poi_state[bracket] = -1.0

        return poi_state

    def rover_scan(self, rovers):
        """
        Rover activates scanner to detect other rovers within the environment
        """
        rover_state = np.zeros(int(360.0 / self.sensor_res))
        temp_rover_dist_list = [[] for _ in range(int(360.0 / self.sensor_res))]

        # Log rover distances into brackets
        for rk in rovers:
            if self.self_id != rovers[rk].self_id:  # Ignore self
                rov_x = rovers[rk].x_pos
                rov_y = rovers[rk].y_pos

                angle, dist = self.get_angle_dist(self.x_pos, self.y_pos, rov_x, rov_y)
                bracket = int(angle / self.sensor_res)
                if bracket > 3:
                    bracket -= 4
                temp_rover_dist_list[bracket].append(1 / dist)

        # Encode Rover information into the state vector
        for bracket in range(int(360 / self.sensor_res)):
            num_rovers_bracket = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
            if num_rovers_bracket > 0:
                if self.sensor_type == 'density':
                    rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_rovers_bracket  # Density Sensor
                elif self.sensor_type == 'summed':
                    rover_state[bracket] = sum(temp_rover_dist_list[bracket])  # Summed Distance Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                rover_state[bracket] = -1.0

        return rover_state

    def get_angle_dist(self, x, y, tx, ty):
        """
        Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        :param tx: X-Position of sensor target
        :param ty: Y-Position of sensor target
        :param x: X-Position of scanning rover
        :param y: Y-Position of scanning rover
        :return: angle, dist
        """

        vx = x - tx
        vy = y - ty

        angle = math.atan2(vy, vx)*(180.0/math.pi)

        while angle < 0.0:
            angle += 360.0
        while angle > 360.0:
            angle -= 360.0
        if math.isnan(angle):
            angle = 0.0

        dist = (vx**2) + (vy**2)

        # Clip distance to not overwhelm activation function in NN
        if dist < self.delta_min:
            dist = self.delta_min

        return angle, dist

    # Motor Control NN ------------------------------------------------------------------------------------------------
    def get_weights(self, nn_weights):
        """
        Apply chosen network weights to the agent's neuro-controller
        """
        self.weights["Layer1"] = np.reshape(np.mat(nn_weights["L1"]), [self.n_hnodes, self.n_inputs])
        self.weights["Layer2"] = np.reshape(np.mat(nn_weights["L2"]), [self.n_outputs, self.n_hnodes])
        self.weights["input_bias"] = np.reshape(np.mat(nn_weights["b1"]), [self.n_hnodes, 1])
        self.weights["hidden_bias"] = np.reshape(np.mat(nn_weights["b2"]), [self.n_outputs, 1])

    def get_nn_outputs(self):
        """
        Run rover NN to generate actions
        """
        self.hidden_layer = np.dot(self.weights["Layer1"], self.input_layer) + self.weights["input_bias"]
        self.hidden_layer = self.sigmoid(self.hidden_layer)

        self.output_layer = np.dot(self.weights["Layer2"], self.hidden_layer) + self.weights["hidden_bias"]
        self.output_layer = self.sigmoid(self.output_layer)

        for i in range(self.n_outputs):
            self.rover_actions[i] = self.output_layer[i, 0]

    # Activation Functions -------------------------------------------------------------------------------------------
    def tanh(self, inp):  # Tanh function as activation function
        """
        tanh neural network activation function
        """

        tanh = (2 / (1 + np.exp(-2 * inp))) - 1

        return tanh

    def sigmoid(self, inp):  # Sigmoid function as activation function
        """
        sigmoid neural network activation function
        """

        sig = 1 / (1 + np.exp(-inp))

        return sig