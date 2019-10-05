import sys
import math
import os
from AADI_RoverDomain.rover_setup import *


class RoverDomain:

    def __init__(self, p):

        # World attributes
        self.world_x = p.x_dim
        self.world_y = p.y_dim
        self.nrovers = p.num_rovers
        self.n_pois = p.num_pois
        self.observation_space = np.zeros((1, int(2 * 360 / p.angle_resolution)))
        self.c_req = p.coupling
        self.min_dist = p.min_distance
        self.obs_radius = p.min_observation_dist
        self.istep = 0  # Current Step counter

        # Rover parameters
        self.create_new_world_config = p.new_world_config
        self.rover_steps = p.num_steps
        self.n_rover_inputs = p.num_inputs
        self.angle_res = p.angle_resolution
        self.rover_sensors = p.sensor_model

        # Rover position vectors
        self.rover_pos = np.zeros((p.num_rovers, 3))
        self.rover_initial_pos = np.zeros((p.num_rovers, 3))

        # Rover path trace for trajectory-wide global reward computation and vizualization purposes
        self.rover_path = np.zeros(((p.num_steps + 1), self.nrovers, 3))

        # POI position and value vectors
        self.poi_pos = np.zeros((p.num_pois, 2))
        self.poi_values = np.zeros(p.num_pois)
        self.poi_rewards = np.zeros(p.num_pois)

    def inital_world_setup(self):
        """
        Set rover starting positions, POI positions and POI values
        :return: none
        """

        if self.create_new_world_config == True:
            # Initialize rover positions
            self.rover_pos = init_rover_pos_random_concentrated(self.nrovers, self.world_x, self.world_y)
            self.rover_initial_pos = self.rover_pos.copy()  # Track initial setup

            # Initialize POI positions and values
            self.poi_pos = init_poi_pos_circle(self.n_pois, self.world_x, self.world_y)
            self.poi_values = init_poi_vals_half_and_half(self.n_pois)
            self.save_world_configuration()
        else:
            # Initialize rover positions
            self.rover_pos = init_rover_pos_txt_file(self.nrovers)
            self.rover_initial_pos = self.rover_pos.copy()  # Track initial setup

            # Initialize POI positions and values
            self.poi_pos = init_poi_positions_txt_file(self.n_pois)
            self.poi_values = init_poi_values_txt_file(self.n_pois)

        self.istep = 0
        self.rover_path = np.zeros(((self.rover_steps + 1), self.nrovers, 3))  # Tracks rover trajectories
        self.poi_rewards = np.zeros(self.n_pois)

        for rover_id in range(self.nrovers):  # Record intial positions in rover path
            self.rover_path[self.istep, rover_id, 0] = self.rover_pos[rover_id, 0]
            self.rover_path[self.istep, rover_id, 1] = self.rover_pos[rover_id, 1]
            self.rover_path[self.istep, rover_id, 2] = self.rover_pos[rover_id, 2]

    def save_world_configuration(self):
        """
        Saves world configuration to a txt files in a folder called Output_Data
        :Output: Three txt files for Rover starting positions, POI postions, and POI values
        """
        dir_name = 'Output_Data/'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        rcoords_name = os.path.join(dir_name, 'Rover_Positions.txt')
        pcoords_name = os.path.join(dir_name, 'POI_Positions.txt')
        pvals_name = os.path.join(dir_name, 'POI_Values.txt')

        rov_coords = open(rcoords_name, 'w')
        for r_id in range(self.nrovers):  # Record initial rover positions to txt file
            rov_coords.write('%f' % self.rover_pos[r_id, 0])
            rov_coords.write('\t')
            rov_coords.write('%f' % self.rover_pos[r_id, 1])
            rov_coords.write('\t')
            rov_coords.write('%f' % self.rover_pos[r_id, 2])
            rov_coords.write('\t')
        rov_coords.write('\n')
        rov_coords.close()

        poi_coords = open(pcoords_name, 'w')
        poi_values = open(pvals_name, 'w')
        for p_id in range(self.n_pois):  # Record POI positions and values
            poi_coords.write('%f' % self.poi_pos[p_id, 0])
            poi_coords.write('\t')
            poi_coords.write('%f' % self.poi_pos[p_id, 1])
            poi_coords.write('\t')
            poi_values.write('%f' % self.poi_values[p_id])
            poi_values.write('\t')
        poi_coords.write('\n')
        poi_values.write('\n')
        poi_coords.close()
        poi_values.close()

    def reset_to_init(self):
        """
        Resets rovers to starting positions (does not alter the world's initial state)
        :return: none
        """
        self.poi_rewards = np.zeros(self.n_pois)
        self.rover_pos = self.rover_initial_pos.copy()
        self.rover_path = np.zeros(((self.rover_steps + 1), self.nrovers, 3))
        self.istep = 0

        for rover_id in range(self.nrovers):  # Record initial positions in rover path
            self.rover_path[self.istep, rover_id, 0] = self.rover_pos[rover_id, 0]
            self.rover_path[self.istep, rover_id, 1] = self.rover_pos[rover_id, 1]
            self.rover_path[self.istep, rover_id, 2] = self.rover_pos[rover_id, 2]

    def step(self, joint_action):
        """
        :param joint_action: np array containing output from NN. Array size is (nrovers, 2)
        :return: Joint state of rovers (NN inputs), Done, and Global Reward
        """
        self.istep += 1
        joint_action = np.clip(joint_action, -1.0, 1.0)

        # Update rover positions
        for rover_id in range(self.nrovers):

            x = joint_action[rover_id, 0]
            y = joint_action[rover_id, 1]
            theta = math.atan(y/x) * (180/math.pi)
            if theta < 0:
                theta += 360
            if theta > 360:
                theta -= 360
            if math.isnan(theta):
                theta = 0.0

            # Update rover position
            self.rover_pos[rover_id, 0] += x
            self.rover_pos[rover_id, 1] += y
            self.rover_pos[rover_id, 2] = theta


        # Update rover path
        for rover_id in range(self.nrovers):
            self.rover_path[self.istep, rover_id, 0] = self.rover_pos[rover_id, 0]
            self.rover_path[self.istep, rover_id, 1] = self.rover_pos[rover_id, 1]
            self.rover_path[self.istep, rover_id, 2] = self.rover_pos[rover_id, 2]

        # Computes done
        done = int(self.istep >= self.rover_steps)

        joint_state = self.get_joint_state()

        # g_reward = self.calc_global_reward_alpha()

        return joint_state, done

    def get_joint_state(self):
        """
        joint_state is an array of size [nrovers][8] containing inputs for NN
        :return: joint_state
        """

        joint_state = np.zeros((self.nrovers, self.n_rover_inputs))

        for rover_id in range(self.nrovers):
            self_x = self.rover_pos[rover_id, 0]; self_y = self.rover_pos[rover_id, 1]
            self_orient = self.rover_pos[rover_id, 2]

            rover_state = [0.0 for _ in range(int(360 / self.angle_res))]
            poi_state = [0.0 for _ in range(int(360 / self.angle_res))]
            temp_poi_dist_list = [[] for _ in range(int(360 / self.angle_res))]
            temp_rover_dist_list = [[] for _ in range(int(360 / self.angle_res))]

            # Log POI distances into brackets
            for poi_id in range(self.n_pois):
                poi_x = self.poi_pos[poi_id, 0]
                poi_y = self.poi_pos[poi_id, 1]
                poi_value = self.poi_values[poi_id]

                angle, dist = self.get_angle_dist(self_x, self_y, poi_x, poi_y)

                if dist >= self.obs_radius:
                    continue  # Observability radius

                angle -= self_orient
                if angle < 0:
                    angle += 360

                bracket = int(angle / self.angle_res)
                if bracket >= len(temp_poi_dist_list):
                    print("ERROR: BRACKET EXCEED LIST", bracket, len(temp_poi_dist_list))
                    bracket = len(temp_poi_dist_list) - 1
                if dist < self.min_dist:  # Clip distance to not overwhelm tanh in NN
                    dist = self.min_dist

                temp_poi_dist_list[bracket].append(poi_value/dist)

            # Log rover distances into brackets
            for other_rover_id in range(self.nrovers):
                if other_rover_id == rover_id: # Ignore self
                    continue
                rov_x = self.rover_pos[other_rover_id, 0]
                rov_y = self.rover_pos[other_rover_id, 1]
                angle, dist = self.get_angle_dist(self_x, self_y, rov_x, rov_y)

                if dist >= self.obs_radius:
                    continue  # Observability radius

                angle -= self_orient
                if angle < 0:
                    angle += 360

                if dist < self.min_dist:  # Clip distance to not overwhelm sigmoid in NN
                    dist = self.min_dist

                bracket = int(angle / self.angle_res)
                if bracket >= len(temp_rover_dist_list):
                    print("ERROR: BRACKET EXCEED LIST", bracket, len(temp_rover_dist_list))
                    bracket = len(temp_rover_dist_list) - 1
                temp_rover_dist_list[bracket].append(1/dist)

            # Encode the information into the state vector
            for bracket in range(int(360 / self.angle_res)):
                # POIs
                num_poi = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
                if num_poi > 0:
                    if self.rover_sensors == 'density':
                        poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi  # Density Sensor
                    elif self.rover_sensors == 'summed':
                        poi_state[bracket] = sum(temp_poi_dist_list[bracket])  # Summed Distance Sensor
                    elif self.rover_sensors == 'closest':
                        poi_state[bracket] = max(temp_poi_dist_list[bracket])  # Closest Sensor
                    else:
                        sys.exit('Incorrect sensor model')
                else:
                    poi_state[bracket] = -1.0
                joint_state[rover_id, bracket] = poi_state[bracket]

                # Rovers
                num_agents = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
                if num_agents > 0:
                    if self.rover_sensors == 'density':
                        rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_agents  # Density Sensor
                    elif self.rover_sensors == 'summed':
                        rover_state[bracket] = sum(temp_rover_dist_list[bracket])  # Summed Distance Sensor
                    elif self.rover_sensors == 'closest':
                        rover_state[bracket] = max(temp_rover_dist_list[bracket])  # Closest Sensor
                    else:
                        sys.exit('Incorrect sensor model')
                else:
                    rover_state[bracket] = -1.0
                joint_state[rover_id, (bracket + 4)] = rover_state[bracket]

        return joint_state


    def get_angle_dist(self, rovx, rovy, x, y):
        """
        Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        :param rovx: X-Position of rover
        :param rovy: Y-Position of rover
        :param x: X-Position of POI or other rover
        :param y: Y-Position of POI or other rover
        :return: angle, dist
        """
        vx = x - rovx; vy = y - rovy
        angle = math.atan(vy/vx)*(180/math.pi)

        if angle < 0:
            angle += 360
        if angle > 360:
            angle -= 360
        if math.isnan(angle):
            angle = 0.0

        dist = math.sqrt((vx * vx) + (vy * vy))

        return angle, dist


    def calc_global(self):
        """
        Calculates global reward for current world state.
        :return: global_reward
        """
        global_reward = 0.0

        for poi_id in range(self.n_pois):
            rover_distances = np.zeros(self.nrovers)
            observer_count = 0

            for agent_id in range(self.nrovers):
                # Calculate distance between agent and POI
                x_distance = self.poi_pos[poi_id, 0] - self.rover_pos[agent_id, 0]
                y_distance = self.poi_pos[poi_id, 1] - self.rover_pos[agent_id, 1]
                distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                if distance < self.min_dist:
                    distance = self.min_dist

                rover_distances[agent_id] = distance

                # Check if agent observes poi and update observer count if true
                if distance < self.obs_radius:
                    observer_count += 1

            # Update global reward if POI is observed
            if observer_count >= self.c_req:
                global_reward += self.poi_values[poi_id]

        return global_reward
