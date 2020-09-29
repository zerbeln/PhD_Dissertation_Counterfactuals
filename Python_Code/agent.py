import numpy as np
import math
import sys
import csv
from parameters import parameters as p


class Rover:
    def __init__(self, rov_id):
        # Rover Parameters
        self.sensor_range = p["obs_rad"]
        self.sensor_readings = np.zeros(p["n_inputs"])
        self.suggestion_inputs = np.zeros(p["n_inputs"])
        self.self_id = rov_id
        self.angle_res = p["angle_res"]
        self.sensor_type = p["sensor_model"]
        self.controller_type = p["ctrl_type"]
        self.rover_actions = np.zeros(p["n_outputs"])
        self.rover_x = 0.0
        self.rover_y = 0.0
        self.rover_theta = 0.0
        self.rx_init = 0.0
        self.ry_init = 0.0
        self.rt_init = 0.0

        # Rover Neuro-Controller -----------------------------------------------------------------------------
        self.n_inputs = p["n_inputs"]
        self.n_outputs = p["n_outputs"]
        self.n_hnodes = p["n_hnodes"]  # Number of nodes in hidden layer

        # Standard Neural Network ----------------------------------------------------------------------------
        self.weights = {}
        self.input_layer = np.reshape(np.mat(np.zeros(self.n_inputs)), [self.n_inputs, 1])
        self.hidden_layer = np.reshape(np.mat(np.zeros(self.n_hnodes)), [self.n_hnodes, 1])
        self.output_layer = np.reshape(np.mat(np.zeros(self.n_outputs)), [self.n_outputs, 1])

        # GRU Controller --------------------------------------------------------------------------------------
        # Memory
        self.mem_block_size = p["mem_block_size"]
        self.mem_block = np.zeros(p["mem_block_size"])
        self.decoded_memory = np.mat(np.zeros(self.mem_block_size))
        self.encoded_memory = np.mat(np.zeros(self.mem_block_size))
        self.decoder_weights = {}
        self.encoder_weights = {}

        # Network Outputs
        self.out_layer = np.mat(np.zeros(self.n_outputs))
        self.prev_out_layer = np.mat(np.zeros(self.n_outputs))
        self.out_layer_weights = np.mat(np.zeros(self.mem_block_size))
        self.out_bias_weights = np.mat(np.zeros(self.n_outputs))

        # Input Gate
        self.igate_weights = {}
        self.igate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Block Input
        self.block_weights = {}
        self.block_input = np.mat(np.zeros(self.mem_block_size))
        self.block_output = np.mat(np.zeros(self.mem_block_size))

        # Read Gate
        self.rgate_weights = {}
        self.rgate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Write Gate
        self.wgate_weights = {}
        self.wgate_outputs = np.mat(np.zeros(self.mem_block_size))

        # Suggestion Gate
        self.sgate_weights = {}
        self.sgate_outputs = np.mat(np.zeros(self.mem_block_size))

    def initialize_rover(self):
        """
        Load initial rover position from saved csvfile
        """
        config_input = []
        with open('Output_Data/Rover_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        self.rx_init = float(config_input[self.self_id][0])
        self.ry_init = float(config_input[self.self_id][1])
        self.rt_init = float(config_input[self.self_id][2])

        self.rover_x = self.rx_init
        self.rover_y = self.ry_init
        self.rover_theta = self.rt_init

    def reset_rover(self):
        """
        Resets the rover to its initial position in the world
        """
        self.rover_x = self.rx_init
        self.rover_y = self.ry_init
        self.rover_theta = self.rt_init
        self.sensor_readings = np.zeros(p["n_inputs"])
        self.suggestion_inputs = np.zeros(p["n_inputs"])
        self.mem_block = np.zeros(p["mem_block_size"])

    def clear_memory(self):
        self.mem_block = np.zeros(p["mem_block_size"])

    def step(self):
        """
        :return: Joint state of rovers (NN inputs), Done, and Global Reward
        """

        joint_action = np.clip(self.rover_actions, -1.0, 1.0)

        # Update rover positions
        x = joint_action[0]
        y = joint_action[1]
        theta = math.atan(y/x) * (180.0/math.pi)

        if theta < 0.0:
            theta += 360.0
        elif theta > 360.0:
            theta -= 360.0
        elif math.isnan(theta):
            theta = 0.0

        # Update rover position
        if 0.0 <= (self.rover_x + x) < (p["x_dim"]-1.0) and 0.0 <= (self.rover_y + y) < (p["y_dim"]-1.0):
            self.rover_x += x
            self.rover_y += y
        self.rover_theta = theta

    def scan_environment(self, rovers, pois, sgst):
        """
        Constructs the state information that gets passed to the rover's neuro-controller
        """
        self.poi_sensor_scan(pois, sgst)
        self.rover_sensor_scan(rovers, sgst)

    def poi_sensor_scan(self, pois, sgst):
        """
        Rover uses POI scanner to detect POIs in each quadrant
        """
        poi_state = np.zeros(int(360.0 / self.angle_res))
        suggestion_state = np.zeros(int(360.0 / self.angle_res))
        temp_poi_dist_list = [[] for _ in range(int(360.0 / self.angle_res))]
        temp_poi_sgst_list = [[] for _ in range(int(360.0 / self.angle_res))]

        # Log POI distances into brackets
        for poi_id in range(p["n_poi"]):
            poi_x = pois[poi_id, 0]
            poi_y = pois[poi_id, 1]
            poi_value = pois[poi_id, 2]

            angle, dist = self.get_angle_dist(self.rover_x, self.rover_y, poi_x, poi_y)
            angle -= self.rover_theta
            if dist < p["min_distance"]:  # Clip distance to not overwhelm tanh in NN
                dist = p["min_distance"]

            bracket = int(angle / self.angle_res)
            if bracket >= len(temp_poi_dist_list):
                bracket = len(temp_poi_dist_list) - 1

            temp_poi_dist_list[bracket].append(poi_value / dist)
            if sgst == "high_val" and poi_value > 5.0:
                temp_poi_sgst_list[bracket].append(poi_value / dist)
            elif sgst == "high_val" and poi_value <= 5.0:
                temp_poi_sgst_list[bracket].append(-10*poi_value / dist)
            elif sgst == "low_val" and poi_value <= 5.0:
                temp_poi_sgst_list[bracket].append(poi_value / dist)
            elif sgst == "low_val" and poi_value > 5.0:
                temp_poi_sgst_list[bracket].append(-10*poi_value / dist)

        # Encode the information into the state vector
        for bracket in range(int(360 / self.angle_res)):
            num_poi = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
            num_poi_double = len(temp_poi_dist_list[bracket])
            if num_poi > 0:
                if self.sensor_type == 'density':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi_double  # Density Sensor
                    suggestion_state[bracket] = sum(temp_poi_sgst_list[bracket]) / num_poi_double
                elif self.sensor_type == 'summed':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket])  # Summed Distance Sensor
                    suggestion_state[bracket] = sum(temp_poi_sgst_list[bracket])
                elif self.sensor_type == 'closest':
                    poi_state[bracket] = max(temp_poi_dist_list[bracket])  # Closest Sensor
                    suggestion_state[bracket] = max(temp_poi_sgst_list[bracket])
                else:
                    sys.exit('Incorrect sensor model')
            else:
                poi_state[bracket] = -1.0
                suggestion_state[bracket] = -1.0

            self.sensor_readings[bracket] = poi_state[bracket]
            self.suggestion_inputs[bracket] = suggestion_state[bracket]

    def rover_sensor_scan(self, rovers, sgst):
        """
        Rover uses rover sensor to detect other rovers in each quadrant
        """
        rover_state = np.zeros(int(360.0 / self.angle_res))
        suggestion_state = np.zeros(int(360.0 / self.angle_res))
        temp_rover_dist_list = [[] for _ in range(int(360.0 / self.angle_res))]
        temp_rover_sgst_list = [[] for _ in range(int(360.0 / self.angle_res))]

        # Log rover distances into brackets
        for rover_id in range(p["n_rovers"]):
            if self.self_id == rover_id: # Ignore self
                continue
            rov_x = rovers["Rover{0}".format(rover_id)].rover_x
            rov_y = rovers["Rover{0}".format(rover_id)].rover_y

            angle, dist = self.get_angle_dist(self.rover_x, self.rover_y, rov_x, rov_y)

            angle -= self.rover_theta

            if dist < p["min_distance"]:  # Clip distance to not overwhelm sigmoid in NN
                dist = p["min_distance"]

            bracket = int(angle / self.angle_res)
            if bracket >= len(temp_rover_dist_list):
                bracket = len(temp_rover_dist_list) - 1
            temp_rover_dist_list[bracket].append(1/dist)
            temp_rover_sgst_list[bracket].append(1/dist)

        # Encode the information into the state vector
        for bracket in range(int(360/self.angle_res)):
            num_agents = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
            num_agents_double = len(temp_rover_dist_list[bracket])
            if num_agents > 0:
                if self.sensor_type == 'density':
                    rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_agents_double  # Density Sensor
                    suggestion_state[bracket] = sum(temp_rover_sgst_list[bracket]) / num_agents_double
                elif self.sensor_type == 'summed':
                    rover_state[bracket] = sum(temp_rover_dist_list[bracket])  # Summed Distance Sensor
                    suggestion_state[bracket] = sum(temp_rover_sgst_list[bracket])
                elif self.sensor_type == 'closest':
                    rover_state[bracket] = max(temp_rover_dist_list[bracket])  # Closest Sensor
                    suggestion_state[bracket] = max(temp_rover_sgst_list[bracket])
                else:
                    sys.exit('Incorrect sensor model')
            else:
                rover_state[bracket] = -1.0
                suggestion_state[bracket] = -1.0

            self.sensor_readings[bracket + 4] = rover_state[bracket]
            self.suggestion_inputs[bracket + 4] = suggestion_state[bracket]

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
        angle = math.atan(vy/vx)*(180.0/math.pi)

        while angle < 0.0:
            angle += 360.0
        while angle > 360.0:
            angle -= 360.0
        if math.isnan(angle):
            angle = 0.0

        dist = math.sqrt((vx**2) + (vy**2))

        return angle, dist

    # Rover Neuro-Controller functions --------------------------------------------------------------------------------
    def run_neuro_controller(self):
        """
        Run the neuro-controller from a single function call
        """
        if self.controller_type == "NN":
            self.get_inputs()
            self.get_nn_outputs()
        elif self.controller_type == "GRU":
            state_vec = self.sensor_readings.copy()
            suggest_vec = self.suggestion_inputs.copy()
            m_block = self.mem_block.copy()
            self.run_input_gate(state_vec, suggest_vec, m_block)
            self.run_suggestion_gate(state_vec, suggest_vec, m_block)
            self.run_read_gate(state_vec, suggest_vec, m_block)
            self.create_block_inputs(state_vec, suggest_vec, m_block)
            self.memory_decoder(m_block)
            self.hidden_activation()
            self.run_write_gate(state_vec, suggest_vec, m_block)
            self.memory_encoder()
            self.update_memory()
            self.prev_out_layer = self.out_layer.copy()

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

    def get_weights(self, weights):
        """
        Apply weights to the neuro-controller
        """
        if self.controller_type == "NN":
            self.get_nn_weights(weights)
        elif self.controller_type == "GRU":
            self.get_gru_weights(weights)

    # Standard NN ------------------------------------------------------------------------------------------------
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

    # GRU ------------------------------------------------------------------------------------------------------------
    def clear_gru_outputs(self):
        """
        Clears the various outputs of gates (sets them to 0)
        :return:
        """
        self.igate_outputs = np.mat(np.zeros(self.mem_block_size))
        self.wgate_outputs = np.mat(np.zeros(self.mem_block_size))
        self.rgate_outputs = np.mat(np.zeros(self.mem_block_size))
        self.block_output = np.mat(np.zeros(self.mem_block_size))
        self.sgate_outputs = np.mat(np.zeros(self.mem_block_size))
        self.out_layer = np.mat(np.zeros(self.n_outputs))
        self.prev_out_layer = np.mat(np.zeros(self.n_outputs))

    def get_gru_weights(self, weights):  # Get weights from CCEA population
        """
        Receive rover NN weights from CCEA
        :return: None
        """

        # Output weights
        self.out_bias_weights = np.mat(weights["b_out"])
        self.out_layer_weights = np.mat(weights["p_out"])

        # Input gate weights
        n_mat = np.mat(weights["n_igate"])
        self.igate_weights["K"] = np.reshape(np.mat(weights["k_igate"]), [self.mem_block_size, self.n_inputs])  # nx8
        self.igate_weights["S"] = np.reshape(np.mat(weights["s_igate"]), [self.mem_block_size, self.n_inputs])
        self.igate_weights["R"] = np.reshape(np.mat(weights["r_igate"]), [self.mem_block_size, self.n_outputs])  # nx2
        self.igate_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.igate_weights["b"] = np.reshape(np.mat(weights["b_igate"]), [self.mem_block_size, 1])  # nx1

        # Block Input
        n_mat = np.mat(weights["n_block"])
        self.block_weights["K"] = np.reshape(np.mat(weights["k_block"]), [self.mem_block_size, self.n_inputs])  # nx8
        self.block_weights["S"] = np.reshape(np.mat(weights["s_block"]), [self.mem_block_size, self.n_inputs])
        self.block_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.block_weights["b"] = np.reshape(np.mat(weights["b_block"]), [self.mem_block_size, 1])  # nx1

        # Read gate weights
        n_mat = np.mat(weights["n_rgate"])
        self.rgate_weights["K"] = np.reshape(np.mat(weights["k_rgate"]), [self.mem_block_size, self.n_inputs])  # nx8
        self.rgate_weights["S"] = np.reshape(np.mat(weights["s_rgate"]), [self.mem_block_size, self.n_inputs])
        self.rgate_weights["R"] = np.reshape(np.mat(weights["r_rgate"]), [self.mem_block_size, self.n_outputs])  # nx2
        self.rgate_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.rgate_weights["b"] = np.reshape(np.mat(weights["b_rgate"]), [self.mem_block_size, 1])  # nx1

        # Write Gate Weights
        n_mat = np.mat(weights["n_wgate"])
        self.wgate_weights["K"] = np.reshape(np.mat(weights["k_wgate"]), [self.mem_block_size, self.n_inputs])  # nx8
        self.wgate_weights["S"] = np.reshape(np.mat(weights["s_wgate"]), [self.mem_block_size, self.n_inputs])
        self.wgate_weights["R"] = np.reshape(np.mat(weights["r_wgate"]), [self.mem_block_size, self.n_outputs])  # nx2
        self.wgate_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.wgate_weights["b"] = np.reshape(np.mat(weights["b_wgate"]), [self.mem_block_size, 1])  # nx1

        # Suggestion Gate Weights
        n_mat = np.mat(weights["n_sgate"])
        self.sgate_weights["K"] = np.reshape(np.mat(weights["k_sgate"]), [self.mem_block_size, self.n_inputs])  # nx8
        self.sgate_weights["S"] = np.reshape(np.mat(weights["s_sgate"]), [self.mem_block_size, self.n_inputs])
        self.sgate_weights["R"] = np.reshape(np.mat(weights["r_sgate"]), [self.mem_block_size, self.n_outputs])  # nx2
        self.sgate_weights["N"] = np.reshape(n_mat, [self.mem_block_size, self.mem_block_size])  # nxn
        self.sgate_weights["b"] = np.reshape(np.mat(weights["b_sgate"]), [self.mem_block_size, 1])  # nx1

        # Memory Weights
        n_mat_dec = np.mat(weights["n_dec"])
        n_mat_enc = np.mat(weights["z_enc"])
        self.decoder_weights["N"] = np.reshape(n_mat_dec, [self.mem_block_size, self.mem_block_size])  # nxn
        self.decoder_weights["b"] = np.reshape(np.mat(weights["b_dec"]), [self.mem_block_size, 1])  # nx1
        self.encoder_weights["Z"] = np.reshape(n_mat_enc, [self.mem_block_size, self.mem_block_size])  # nxn
        self.encoder_weights["b"] = np.reshape(np.mat(weights["b_enc"]), [self.mem_block_size, 1])  # nx1

    def run_input_gate(self, state_vec, suggest_vec, mem_block):
        """
        Process sensor inputs through input gate
        :param mem_block:
        :param state_vec:
        :return:
        """

        x = np.reshape(np.mat(state_vec), [self.n_inputs, 1])  # 8x1
        y = np.reshape(np.mat(self.prev_out_layer), [self.n_outputs, 1])  # 2x1
        z = np.reshape(np.mat(suggest_vec), [self.n_inputs, 1])  # 8x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.igate_weights["b"]  # nx1

        Ki_x = np.dot(self.igate_weights["K"], x)  # nx8 * 8x1 = nx1
        Ri_y = np.dot(self.igate_weights["R"], y)  # nx2 * 2x1 = nx1
        Ni_m = np.dot(self.igate_weights["N"], m)  # nxn * nx1 = nx1
        Si_z = np.dot(self.igate_weights["S"], z)

        self.igate_outputs = Ki_x + Ri_y + Ni_m + + Si_z + b  # nx1
        for i in range(self.mem_block_size):
            self.igate_outputs[i, 0] = self.sigmoid(self.igate_outputs[i, 0])

    def run_read_gate(self, state_vec, suggest_vec, mem_block):
        """
        Process read gate information
        :param mem_block:
        :param state_vec:
        :return:
        """

        x = np.reshape(np.mat(state_vec), [self.n_inputs, 1])  # 8x1
        y = np.reshape(np.mat(self.prev_out_layer), [self.n_outputs, 1])  # 2x1
        z = np.reshape(np.mat(suggest_vec), [self.n_inputs, 1])  # 8x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.rgate_weights["b"]  # nx1

        Kr_x = np.dot(self.rgate_weights["K"], x)  # nx8 * 8x1
        Rr_y = np.dot(self.rgate_weights["R"], y)  # nx2 * 2x1
        Nr_m = np.dot(self.rgate_weights["N"], m)  # nxn * 1x1
        Si_z = np.dot(self.rgate_weights["S"], z)

        self.rgate_outputs = Kr_x + Rr_y + Nr_m + Si_z + b  # nx1
        for i in range(self.mem_block_size):
            self.rgate_outputs[i, 0] = self.sigmoid(self.rgate_outputs[i, 0])

    def run_suggestion_gate(self, state_vec, suggest_vec, mem_block):
        """
        Process suggestion gate information
        :param mem_block:
        :param suggest_vec:
        :return:
        """
        x = np.reshape(np.mat(state_vec), [self.n_inputs, 1])  # 8x1
        y = np.reshape(np.mat(self.prev_out_layer), [self.n_outputs, 1])  # 2x1
        z = np.reshape(np.mat(suggest_vec), [self.n_inputs, 1])  # 8x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.sgate_weights["b"]  # nx1

        Kr_x = np.dot(self.sgate_weights["K"], x)  # nx8 * 8x1
        Rr_y = np.dot(self.sgate_weights["R"], y)  # nx2 * 2x1
        Nr_m = np.dot(self.sgate_weights["N"], m)  # nxn * 1x1
        Si_z = np.dot(self.sgate_weights["S"], z)



        self.sgate_outputs = Kr_x + Rr_y + Nr_m + Si_z + b  # nx1
        for i in range(self.mem_block_size):
            self.sgate_outputs[i, 0] = self.sigmoid(self.sgate_outputs[i, 0])

    def run_write_gate(self, state_vec, suggest_vec, mem_block):
        """
        Process write gate
        :param mem_block:
        :param state_vec:
        :return:
        """

        x = np.reshape(np.mat(state_vec), [self.n_inputs, 1])  # 8x1
        y = np.reshape(np.mat(self.prev_out_layer), [self.n_outputs, 1])  # 2x1
        z = np.reshape(np.mat(suggest_vec), [self.n_inputs, 1])  # 8x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.wgate_weights["b"]  # nx1

        Kw_x = np.dot(self.wgate_weights["K"], x)  # nx8 * 8x1
        Rw_x = np.dot(self.wgate_weights["R"], y)  # nx2 * 2x1
        Nw_x = np.dot(self.wgate_weights["N"], m)  # nxn * nx1
        Si_z = np.dot(self.wgate_weights["S"], z)

        self.wgate_outputs = Kw_x + Rw_x + Nw_x + Si_z + b  # nx1
        for i in range(self.mem_block_size):
            self.wgate_outputs[i, 0] = self.sigmoid(self.wgate_outputs[i, 0])

    def create_block_inputs(self, state_vec, suggest_vec, mem_block):
        """
        Create the input layer for the block (feedforward network)
        :param mem_block:
        :param state_vec:
        :return:
        """

        x = np.reshape(np.mat(state_vec), [self.n_inputs, 1])  # 8x1
        z = np.reshape(np.mat(suggest_vec), [self.n_inputs, 1])  # 8x1
        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])  # nx1
        b = self.block_weights["b"]  # nx1

        Kp_x = np.dot(self.block_weights["K"], x)  # nx8 * 8x1
        Np_m = np.dot(self.block_weights["N"], m)  # nxn * nx1
        Si_z = np.dot(self.block_weights["S"], z)

        self.block_input = Kp_x + Np_m + Si_z + b  # nx1
        for i in range(self.mem_block_size):
            self.block_input[i, 0] = self.sigmoid(self.block_input[i, 0])

    def memory_decoder(self, mem_block):
        """
        Decode memory for hidden activation
        :param mem_block:
        :return:
        """

        m = np.reshape(np.mat(mem_block), [self.mem_block_size, 1])
        Nd_m = np.dot(self.decoder_weights["N"], m)  # nxn * nx1
        b = self.decoder_weights["b"]  # nx1

        self.decoded_memory = Nd_m + b  # nx1
        for i in range(self.mem_block_size):
            self.decoded_memory[i, 0] = self.tanh(self.decoded_memory[i, 0])

    def memory_encoder(self):
        """
        Encode memory for update
        :return:
        """
        b = self.encoder_weights["b"]  # nx1
        Z_h = np.dot(self.encoder_weights["Z"], self.block_output)  # nxn * nx1

        self.encoded_memory = Z_h + b  # nx1
        for i in range(self.mem_block_size):
            self.encoded_memory[i, 0] = self.tanh(self.encoded_memory[i, 0])

    def hidden_activation(self):
        """
        Run NN to receive rover action outputs
        :return: None
        """

        r_d = np.multiply(self.rgate_outputs, self.decoded_memory)  # nx1
        p_i = np.multiply(self.block_input, self.igate_outputs)  # nx1
        s_i = np.multiply(self.block_input, self.sgate_outputs)  # nx1

        self.block_output = r_d + p_i + s_i  # nx1

        self.out_layer = np.dot(self.out_layer_weights, self.block_output) + self.out_bias_weights  # 1x1

        for v in range(self.n_outputs):
            self.out_layer[0, v] = self.tanh(self.out_layer[0, v])
            self.rover_actions[v] = self.out_layer[0, v]

    def update_memory(self):
        """
        GRU-MB agent updates the stored memory
        :return:
        """
        alpha = 0.1
        wgate = np.reshape(self.wgate_outputs, [1, self.mem_block_size])
        enc_mem = np.reshape(self.encoded_memory, [1, self.mem_block_size])

        var1 = (1 - alpha) * (self.mem_block + np.multiply(wgate, enc_mem))
        var2 = alpha * (np.multiply(wgate, enc_mem) + np.multiply((1 - wgate), self.mem_block))

        self.mem_block = var1 + var2
