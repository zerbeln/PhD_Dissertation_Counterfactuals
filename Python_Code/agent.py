import numpy as np
import math
import sys
import csv
from parameters import parameters as p


class Rover:
    def __init__(self, rov_id, n_inp=8, n_out=2, n_hid=9):
        # Rover Parameters
        self.sensor_type = 'density'  # Type of sesnors rover is equipped with
        self.sensor_range = p["observation_radius"]  # Distances which sensors can observe POI
        self.sensor_res = p["angle_res"]  # Angular resolution of the sensors
        self.sensor_readings = np.zeros(n_inp)  # Number of sensor inputs for Neural Network
        self.poi_distances = np.ones(p["n_poi"]) * 1000.00  # Records distances measured from sensors
        self.self_id = rov_id
        self.rover_actions = np.zeros(n_out)  # Motor actions from neural network outputs
        self.initial_pos = np.zeros(3)
        self.pos = np.zeros(3)

        # Suggestion Interpretation Parameters ---------------------------------------------------------------
        self.n_suggestions = p["n_suggestions"]
        self.alpha = 0.3
        self.policy_belief = np.ones(self.n_suggestions) * 0.5

        # Rover Motor Controller -----------------------------------------------------------------------------
        self.n_inputs = n_inp
        self.n_outputs = n_out
        self.n_hnodes = n_hid  # Number of nodes in hidden layer
        self.weights = {}
        self.input_layer = np.reshape(np.mat(np.zeros(self.n_inputs)), [self.n_inputs, 1])
        self.hidden_layer = np.reshape(np.mat(np.zeros(self.n_hnodes)), [self.n_hnodes, 1])
        self.output_layer = np.reshape(np.mat(np.zeros(self.n_outputs)), [self.n_outputs, 1])

    def initialize_rover(self, srun):
        """
        Load initial rover position from saved csvfile
        """
        config_input = []
        with open('Output_Data/SRUN{0}/Rover_Config.csv'.format(srun)) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        self.initial_pos[0] = float(config_input[self.self_id][0])
        self.initial_pos[1] = float(config_input[self.self_id][1])
        self.initial_pos[2] = float(config_input[self.self_id][2])

        self.pos = self.initial_pos.copy()

    def reset_rover(self):
        """
        Resets the rover to its initial position in the world
        """
        self.pos = self.initial_pos.copy()
        self.sensor_readings = np.zeros(self.n_inputs)
        self.policy_belief = np.ones(self.n_suggestions) * 0.5
        self.poi_distances = np.ones(p["n_poi"]) * 1000.00

    def update_policy_belief(self, selection_output):
        """
        Update agent's belief about which policy is best based on output from suggestion network
        """

        for s_id in range(self.n_suggestions):
            self.policy_belief[s_id] = self.policy_belief[s_id] + self.alpha*(selection_output[s_id])

    def step(self, x_lim, y_lim):
        """
        Rover executes current actions provided by neuro-controller
        :param x_lim: Outter x-limit of the environment
        :param y_lim:  Outter y-limit of the environment
        :return:
        """
        # Get outputs from neuro-controller
        self.run_neuro_controller()
        rover_action = self.output_layer.copy()
        rover_action = np.clip(rover_action, -1.0, 1.0)

        # Update rover positions based on outputs and assign to dummy variables
        x = rover_action[0, 0] + self.pos[0]
        y = rover_action[1, 0] + self.pos[1]
        theta = math.atan(y / x) * (180.0 / math.pi)

        # Keep theta between 0 and 360 degrees
        while theta < 0.0:
            theta += 360.0
        while theta > 360.0:
            theta -= 360.0
        if math.isnan(theta):
            theta = 0.0

        self.pos[0] = x
        self.pos[1] = y
        self.pos[2] = theta

    def scan_environment(self, rovers, pois, n_rovers):
        """
        Constructs the state information that gets passed to the rover's neuro-controller
        """
        poi_state = self.poi_scan(pois)
        rover_state = self.rover_scan(rovers, n_rovers)

        for i in range(4):
            self.sensor_readings[i] = poi_state[i]
            self.sensor_readings[4 + i] = rover_state[i]

    def poi_scan(self, poi_info):
        """
        Rover queries scanner that detects POIs
        :param poi_info: multi-dimensional numpy array containing coordinates and values of POIs
        :param n_poi: parameter designating the number of POI in the simulation
        :return: Portion of state-vector constructed from POI scanner
        """
        poi_state = np.zeros(int(360.0 / self.sensor_res))
        temp_poi_dist_list = [[] for _ in range(int(360.0 / self.sensor_res))]

        # Log POI distances into brackets
        n_poi = len(poi_info)
        for poi_id in range(n_poi):
            poi_x = poi_info[poi_id, 0]
            poi_y = poi_info[poi_id, 1]
            poi_value = poi_info[poi_id, 2]

            angle, dist = self.get_angle_dist(self.pos[0], self.pos[1], poi_x, poi_y)

            # Clip distance to not overwhelm activation function in NN
            if dist < 1.0:
                dist = 1.0

            self.poi_distances[poi_id] = dist  # Record distance for sensor information
            bracket = int(angle / self.sensor_res)
            temp_poi_dist_list[bracket].append(poi_value / dist)

        # Encode POI information into the state vector
        for bracket in range(int(360 / self.sensor_res)):
            num_poi_bracket = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
            if num_poi_bracket > 0:
                if self.sensor_type == 'density':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi_bracket  # Density Sensor
                elif self.sensor_type == 'summed':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket])  # Summed Distance Sensor
                elif self.sensor_type == 'closest':
                    poi_state[bracket] = max(temp_poi_dist_list[bracket])  # Closest Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                poi_state[bracket] = -1.0

        return poi_state

    def rover_scan(self, rovers, n_rovers):
        """
        Rover activates scanner to detect other rovers within the environment
        :param rovers: Dictionary containing rover positions
        :param n_rovers: Parameter designating the number of rovers in the simulation
        :return: Portion of the state vector created from rover scanner
        """
        rover_state = np.zeros(int(360.0 / self.sensor_res))
        temp_rover_dist_list = [[] for _ in range(int(360.0 / self.sensor_res))]

        # Log rover distances into brackets
        for rover_id in range(n_rovers):
            if self.self_id != rover_id:  # Ignore self
                rov_x = rovers["Rover{0}".format(rover_id)].pos[0]
                rov_y = rovers["Rover{0}".format(rover_id)].pos[1]

                angle, dist = self.get_angle_dist(self.pos[0], self.pos[1], rov_x, rov_y)

                # Clip distance to not overwhelm activation function in NN
                if dist < 1.0:
                    dist = 1.0

                bracket = int(angle / self.sensor_res)
                temp_rover_dist_list[bracket].append(1 / dist)

                # Encode Rover information into the state vector
                num_rovers_bracket = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
                if num_rovers_bracket > 0:
                    if self.sensor_type == 'density':
                        rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_rovers_bracket  # Density Sensor
                    elif self.sensor_type == 'summed':
                        rover_state[bracket] = sum(temp_rover_dist_list[bracket])  # Summed Distance Sensor
                    elif self.sensor_type == 'closest':
                        rover_state[bracket] = max(temp_rover_dist_list[bracket])  # Closest Sensor
                    else:
                        sys.exit('Incorrect sensor model')
                else:
                    rover_state[bracket] = -1.0

        return rover_state

    def get_angle_dist(self, rovx, rovy, x, y):
        """
        Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        :param rovx: X-Position of rover
        :param rovy: Y-Position of rover
        :param x: X-Position of POI or other rover
        :param y: Y-Position of POI or other rover
        :return: angle, dist
        """

        vx = x - rovx
        vy = y - rovy
        assert(vx != 0)
        angle = math.atan(vy/vx)*(180.0/math.pi)

        while angle < 0.0:
            angle += 360.0
        while angle > 360.0:
            angle -= 360.0
        if math.isnan(angle):
            angle = 0.0

        dist = math.sqrt((vx**2) + (vy**2))

        return angle, dist

    # Motor Control NN ------------------------------------------------------------------------------------------------
    def run_neuro_controller(self):
        """
        Run the neuro-controller from a single function call
        """
        self.get_inputs()
        self.get_nn_outputs()

    def get_weights(self, weights):
        """
        Apply weights to the neuro-controller
        """
        self.get_nn_weights(weights)

    def get_inputs(self):  # Get inputs from state-vector
        """
        Transfer state information to the neuro-controller
        :return:
        """

        for i in range(self.n_inputs):
            self.input_layer[i, 0] = self.sensor_readings[i]

    def get_nn_weights(self, nn_weights):
        """
        Apply chosen network weights to the agent's neuro-controller
        :param nn_weights: Dictionary of network weights received from the CCEA
        :return:
        """
        self.weights["Layer1"] = np.reshape(np.mat(nn_weights["L1"]), [self.n_hnodes, self.n_inputs])
        self.weights["Layer2"] = np.reshape(np.mat(nn_weights["L2"]), [self.n_outputs, self.n_hnodes])
        self.weights["input_bias"] = np.reshape(np.mat(nn_weights["b1"]), [self.n_hnodes, 1])
        self.weights["hidden_bias"] = np.reshape(np.mat(nn_weights["b2"]), [self.n_outputs, 1])

    def get_nn_outputs(self):
        """
        Run NN to generate outputs
        :return:
        """
        self.hidden_layer = np.dot(self.weights["Layer1"], self.input_layer) + self.weights["input_bias"]
        for i in range(self.n_hnodes):
            self.hidden_layer[i, 0] = self.tanh(self.hidden_layer[i, 0])

        self.output_layer = np.dot(self.weights["Layer2"], self.hidden_layer) + self.weights["hidden_bias"]
        for i in range(self.n_outputs):
            self.output_layer[i, 0] = self.tanh(self.output_layer[i, 0])
            self.rover_actions[i] = self.output_layer[i, 0]

    # Activation Functions -------------------------------------------------------------------------------------------
    def tanh(self, inp):  # Tanh function as activation function
        """
        tanh neural network activation function
        :param inp: Node value before activation
        :return: Node value after activation
        """
        tanh = (2 / (1 + np.exp(-2 * inp))) - 1

        return tanh

    def sigmoid(self, inp):  # Sigmoid function as activation function
        """
        sigmoid neural network activation function
        :param inp: Node value before activation
        :return: Node value after activation
        """
        sig = 1 / (1 + np.exp(-inp))

        return sig
