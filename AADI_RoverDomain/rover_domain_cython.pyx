import sys
import math
import os
import cython
from AADI_RoverDomain.rover_setup import *


cdef class RoverDomain:
    cdef double world_x, world_y, min_dist, obs_radius, angle_res
    cdef int create_new_world_config
    cdef str rover_sensors
    cdef int nrovers, n_pois, c_req, rover_steps, n_rover_inputs
    cdef public int istep

    cdef public double [:, :] rover_pos
    cdef double [:, :] rover_initial_pos
    cdef public double [:, :, :] rover_path
    cdef public double [:, :] poi_pos
    cdef public double [:] poi_values

    def __cinit__(self, object p):

        # World attributes
        self.world_x = p.x_dim
        self.world_y = p.y_dim
        self.nrovers = int(p.num_rovers)
        self.n_pois = int(p.num_pois)
        self.c_req = int(p.coupling)
        self.min_dist = p.min_distance
        self.obs_radius = p.min_observation_dist
        self.istep = 0  # Current Step counter

        # Rover parameters
        self.create_new_world_config = int(p.new_world_config)
        self.rover_steps = int(p.num_steps)
        self.n_rover_inputs = int(p.num_inputs)
        self.angle_res = p.angle_resolution
        self.rover_sensors = p.sensor_model

        # Rover position vectors
        self.rover_pos = np.zeros((self.nrovers, 3))
        self.rover_initial_pos = np.zeros((self.nrovers, 3))

        # Rover path trace for trajectory-wide global reward computation and vizualization purposes
        self.rover_path = np.zeros(((self.rover_steps + 1), self.nrovers, 3))

        # POI position and value vectors
        self.poi_pos = np.zeros((self.n_pois, 2))
        self.poi_values = np.zeros(self.n_pois)

    cpdef inital_world_setup(self):
        """
        Set rover starting positions, POI positions and POI values
        :return: none
        """

        cdef int rover_id

        if self.create_new_world_config == 1:
            # Initialize rover positions
            self.rover_pos = init_rover_pos_bottom_center(self.nrovers, self.world_x, self.world_y)
            self.rover_initial_pos = self.rover_pos.copy()  # Track initial setup

            # Initialize POI positions and values
            self.poi_pos = init_poi_pos_clusters(self.n_pois, self.world_x, self.world_y)
            self.poi_values = init_poi_vals_clusters(self.n_pois)
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

        for rover_id in range(self.nrovers):  # Record intial positions in rover path
            self.rover_path[self.istep, rover_id, 0] = self.rover_pos[rover_id, 0]
            self.rover_path[self.istep, rover_id, 1] = self.rover_pos[rover_id, 1]
            self.rover_path[self.istep, rover_id, 2] = self.rover_pos[rover_id, 2]

    cpdef save_world_configuration(self):
        """
        Saves world configuration to a txt files in a folder called Output_Data
        :Output: Three txt files for Rover starting positions, POI postions, and POI values
        """
        cdef str dir_name, rcoords_name, pcoords_name, pvals_name
        cdef int r_id, p_id

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

    cpdef reset_to_init(self):
        """
        Resets rovers to starting positions (does not alter the world's initial state)
        :return: none
        """
        cdef int rover_id

        self.rover_pos = self.rover_initial_pos.copy()
        self.rover_path = np.zeros(((self.rover_steps + 1), self.nrovers, 3))
        self.istep = 0

        for rover_id in range(self.nrovers):  # Record initial positions in rover path
            self.rover_path[self.istep, rover_id, 0] = self.rover_pos[rover_id, 0]
            self.rover_path[self.istep, rover_id, 1] = self.rover_pos[rover_id, 1]
            self.rover_path[self.istep, rover_id, 2] = self.rover_pos[rover_id, 2]

    cpdef step(self, double [:, :] joint_action):
        """
        :param joint_action: np array containing output from NN. Array size is (nrovers, 2)
        :return: Joint state of rovers (NN inputs), Done, and Global Reward
        """
        cdef double x, y, theta
        cdef int rover_id, done

        cdef double [:, :] joint_state

        self.istep += 1
        joint_action = np.clip(joint_action, -1.0, 1.0)

        # Update rover positions
        for rover_id in range(self.nrovers):

            x = joint_action[rover_id, 0]
            y = joint_action[rover_id, 1]
            theta = math.atan(y/x) * (180.0/math.pi)
            if theta < 0.0:
                theta += 360.0
            if theta > 360.0:
                theta -= 360.0
            if math.isnan(theta):
                theta = 0.0

            # Update rover position
            self.rover_pos[rover_id, 0] += x
            if self.rover_pos[rover_id, 0] > (self.world_x-1.0):
                self.rover_pos[rover_id, 0] = (self.world_x-1.0)
            if self.rover_pos[rover_id, 0] < 0.0:
                self.rover_pos[rover_id, 0] = 0.0
            self.rover_pos[rover_id, 1] += y
            if self.rover_pos[rover_id, 1] > (self.world_y-1.0):
                self.rover_pos[rover_id, 1] = (self.world_y-1.0)
            if self.rover_pos[rover_id, 1] < 0.0:
                self.rover_pos[rover_id, 1] = 0.0
            self.rover_pos[rover_id, 2] = theta


        # Update rover path
        for rover_id in range(self.nrovers):
            self.rover_path[self.istep, rover_id, 0] = self.rover_pos[rover_id, 0]
            self.rover_path[self.istep, rover_id, 1] = self.rover_pos[rover_id, 1]
            self.rover_path[self.istep, rover_id, 2] = self.rover_pos[rover_id, 2]

        # Computes done
        if self.istep >= self.rover_steps:
            done = 1
        else:
            done = 0

        joint_state = self.get_joint_state()

        return joint_state, done

    cpdef get_joint_state(self):
        """
        joint_state is an array of size [nrovers][8] containing inputs for NN
        :return: joint_state
        """

        cdef double self_x, self_y, self_orient, poi_x, poi_y, poi_value, angle, dist, rov_x, rov_y
        cdef double num_poi_double, num_agents_double
        cdef int rover_id, poi_id, bracket, other_rover_id, num_agents
        cdef double [:, :] joint_state = np.zeros((self.nrovers, self.n_rover_inputs))
        cdef double [:] rover_state, poi_state
        cdef list temp_poi_dist_list, temp_rover_dist_list

        for rover_id in range(self.nrovers):
            self_x = self.rover_pos[rover_id, 0]
            self_y = self.rover_pos[rover_id, 1]
            self_orient = self.rover_pos[rover_id, 2]

            rover_state = np.zeros(int(360.0 / self.angle_res))
            poi_state = np.zeros(int(360.0 / self.angle_res))
            temp_poi_dist_list = [[] for _ in range(int(360.0 / self.angle_res))]
            temp_rover_dist_list = [[] for _ in range(int(360.0 / self.angle_res))]

            # Log POI distances into brackets
            for poi_id in range(self.n_pois):
                poi_x = self.poi_pos[poi_id, 0]
                poi_y = self.poi_pos[poi_id, 1]
                poi_value = self.poi_values[poi_id]

                angle, dist = self.get_angle_dist(self_x, self_y, poi_x, poi_y)

                # if dist >= self.obs_radius:
                #     continue  # Observability radius

                angle -= self_orient
                if angle > 360.0:
                    angle -= 360.0
                if angle < 0.0:
                    angle += 360.0

                bracket = int(angle / self.angle_res)
                if bracket >= len(temp_poi_dist_list):
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

                # if dist >= self.obs_radius:
                #     continue  # Observability radius

                angle -= self_orient
                if angle > 360.0:
                    angle -= 360.0
                if angle < 0.0:
                    angle += 360.0

                if dist < self.min_dist:  # Clip distance to not overwhelm sigmoid in NN
                    dist = self.min_dist

                bracket = int(angle / self.angle_res)
                if bracket >= len(temp_rover_dist_list):
                    bracket = len(temp_rover_dist_list) - 1
                temp_rover_dist_list[bracket].append(1/dist)

            # Encode the information into the state vector
            for bracket in range(int(360 / self.angle_res)):
                # POIs
                num_poi = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
                num_poi_double = len(temp_poi_dist_list[bracket])
                if num_poi > 0:
                    if self.rover_sensors == 'density':
                        poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi_double  # Density Sensor
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
                num_agents_double = len(temp_rover_dist_list[bracket])
                if num_agents > 0:
                    if self.rover_sensors == 'density':
                        rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_agents_double  # Density Sensor
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


    @cython.cdivision(True)
    cpdef get_angle_dist(self, double rovx, double rovy, double x, double y):
        """
        Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        :param rovx: X-Position of rover
        :param rovy: Y-Position of rover
        :param x: X-Position of POI or other rover
        :param y: Y-Position of POI or other rover
        :return: angle, dist
        """
        cdef double vx, vy, angle, dist

        vx = x - rovx; vy = y - rovy
        angle = math.atan(vy/vx)*(180.0/math.pi)

        if angle < 0.0:
            angle += 360.0
        if angle > 360.0:
            angle -= 360.0
        if math.isnan(angle):
            angle = 0.0

        dist = math.sqrt((vx * vx) + (vy * vy))

        return angle, dist

    cpdef poi_val_change(self):
        self.poi_values[0] = 0.0
        self.poi_values[1] = 4.0


    cpdef calc_global(self):
        """
        Calculates global reward for current world state.
        :return: global_reward
        """
        cdef double global_reward = 0.0
        cdef int poi_id, observer_count, agent_id
        cdef double x_distance, y_distance, distance
        cdef double [:] rover_distances

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
