import numpy as np
import random
import math
import sys
import csv
from parameters import parameters as p


class Rover:
    def __init__(self, rov_id):
        self.sensor_range = p["min_obs_dist"]
        self.sensor_readings = np.zeros(p["n_inputs"])
        self.self_id = rov_id
        self.max_steps = p["n_steps"]
        self.angle_res = p["angle_res"]
        self.sensor_type = p["sensor_model"]
        self.mem_block_size = p["mem_block_size"]
        self.mem_block = np.zeros(p["mem_block_size"])

        # User Defined Parameters:
        self.rover_suggestions = np.zeros((p["n_poi"], p["n_steps"]+1))

        # Initialization function
        if p["new_world_config"] == 1:
            self.init_rover_pos_random_concentrated()
            self.rover_x = self.rx_init
            self.rover_y = self.ry_init
            self.rover_theta = self.rt_init
        else:
            self.use_saved_rover_config()
            self.rover_x = self.rx_init
            self.rover_y = self.ry_init
            self.rover_theta = self.rt_init

    def reset_rover(self):
        self.rover_x = self.rx_init
        self.rover_y = self.ry_init
        self.rover_theta = self.rt_init

    def use_saved_rover_config(self):
        config_input = []
        with open('Output_Data/Rover_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        self.rx_init = float(config_input[self.self_id][0])
        self.ry_init = float(config_input[self.self_id][1])
        self.rt_init = float(config_input[self.self_id][2])

    def init_rover_pos_fixed_middle(self):  # Set rovers to fixed starting position
        self.rx_init = 0.5*p["x_dim"]
        self.ry_init = 0.5*p["y_dim"]
        self.rt_init = random.uniform(0.0, 360.0)

    def init_rover_pos_random(self):  # Randomly set rovers on map
        """
        Rovers given random starting positions and orientations
        :return: rover_positions: np array of size (self.n_rovers, 3)
        """

        self.rx_init = random.uniform(0.0, p["x_dim"]-1.0)
        self.ry_init = random.uniform(0.0, p["y_dim"]-1.0)
        self.rt_init = random.uniform(0.0, 360.0)


    def init_rover_pos_random_concentrated(self):
        """
        Rovers given random starting positions within a radius of the center. Starting orientations are random
        :return: rover_positions: np array of size (self.n_rovers, 3)
        """
        radius = 4.0
        center_x = p["x_dim"]/2.0
        center_y = p["y_dim"]/2.0

        x = random.uniform(0.0, p["x_dim"]-1.0)  # Rover X-Coordinate
        y = random.uniform(0.0, p["y_dim"]-1.0)  # Rover Y-Coordinate

        while x > (center_x + radius) or x < (center_x - radius):
            x = random.uniform(0.0, p["x_dim"]-1.0)  # Rover X-Coordinate
        while y > (center_y + radius) or y < (center_y - radius):
            y = random.uniform(0.0, p["y_dim"]-1.0)  # Rover Y-Coordinate

        self.rx_init = x  # Rover X-Coordinate
        self.ry_init = y  # Rover Y-Coordinate
        self.rt_init = random.uniform(0.0, 360.0)  # Rover orientation

    def update_memory(self, nn_wgate, nn_encoded_mem):
        """
        GRU-MB agent updates the stored memory
        :param nn_wgate:
        :param nn_encoded_mem:
        :return:
        """
        alpha = 0.1
        wgate = np.reshape(nn_wgate, [1, self.mem_block_size])
        enc_mem = np.reshape(nn_encoded_mem, [1, self.mem_block_size])

        var1 = (1 - alpha) * (self.mem_block + np.multiply(wgate, enc_mem))
        var2 = alpha * (np.multiply(wgate, enc_mem) + np.multiply((1 - wgate), self.mem_block))

        self.mem_block = var1 + var2

    def init_rover_pos_bottom_center(self):
        self.rx_init = random.uniform((p["x_dim"]/2.0) - 5.0, (p["x_dim"]/2.0) + 5.0)
        self.ry_init = random.uniform(0.0, 2.0)
        self.rt_init = random.uniform(0.0, 360.0)  # Rover orientation

    def step(self, joint_action):
        """
        :param joint_action: np array containing output from NN. Array size is (nrovers, 2)
        :return: Joint state of rovers (NN inputs), Done, and Global Reward
        """

        joint_action = np.clip(joint_action, -1.0, 1.0)

        # Update rover positions
        x = joint_action[0, 0]
        y = joint_action[0, 1]
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

    def scan_environment(self, rovers, pois):
        self.poi_sensor_scan(pois)
        self.rover_sensor_scan(rovers)

    def poi_sensor_scan(self, pois):
        poi_state = np.zeros(int(360.0 / self.angle_res))
        temp_poi_dist_list = [[] for _ in range(int(360.0 / self.angle_res))]

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

        # Encode the information into the state vector
        for bracket in range(int(360 / self.angle_res)):
            num_poi = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
            num_poi_double = len(temp_poi_dist_list[bracket])
            if num_poi > 0:
                if self.sensor_type == 'density':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi_double  # Density Sensor
                elif self.sensor_type == 'summed':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket])  # Summed Distance Sensor
                elif self.sensor_type == 'closest':
                    poi_state[bracket] = max(temp_poi_dist_list[bracket])  # Closest Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                poi_state[bracket] = -1.0
            self.sensor_readings[bracket] = poi_state[bracket]

    def rover_sensor_scan(self, rovers):
        rover_state = np.zeros(int(360.0 / self.angle_res))
        temp_rover_dist_list = [[] for _ in range(int(360.0 / self.angle_res))]

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

        # Encode the information into the state vector
        for bracket in range(int(360/self.angle_res)):
            num_agents = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
            num_agents_double = len(temp_rover_dist_list[bracket])
            if num_agents > 0:
                if self.sensor_type == 'density':
                    rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_agents_double  # Density Sensor
                elif self.sensor_type == 'summed':
                    rover_state[bracket] = sum(temp_rover_dist_list[bracket])  # Summed Distance Sensor
                elif self.sensor_type == 'closest':
                    rover_state[bracket] = max(temp_rover_dist_list[bracket])  # Closest Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                rover_state[bracket] = -1.0
            self.sensor_readings[bracket + 4] = rover_state[bracket]

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
