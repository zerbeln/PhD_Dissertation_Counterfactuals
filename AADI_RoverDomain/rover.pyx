import numpy as np
import random
import math
import sys
import cython
import csv

cdef class Rover:
    cdef double rx_init, ry_init, rt_init, sensor_range, angle_res
    cdef public double rover_x, rover_y, rover_theta
    cdef int max_steps, self_id
    cdef public double [:] sensor_readings
    cdef str sensor_type

    # User Defined Variables:
    cdef public double [:, :] rover_suggestions

    def __cinit__(self, object p, rov_id):
        self.sensor_range = p.min_observation_dist
        self.sensor_readings = np.zeros(8)
        self.self_id = rov_id
        self.max_steps = p.num_steps
        self.angle_res = p.angle_resolution
        self.sensor_type = p.sensor_model

        # User Defined Parameters:
        self.rover_suggestions = np.zeros((p.num_poi, p.num_steps))

        # Initialization function
        if p.new_world_config == 1:
            self.init_rover_pos_random_concentrated(p.x_dim, p.y_dim)
            self.rover_x = self.rx_init
            self.rover_y = self.ry_init
            self.rover_theta = self.rt_init
        else:
            self.use_saved_rover_config(p.num_rovers)
            self.rover_x = self.rx_init
            self.rover_y = self.ry_init
            self.rover_theta = self.rt_init

    cpdef reset_rover(self):
        self.rover_x = self.rx_init
        self.rover_y = self.ry_init
        self.rover_theta = self.rt_init

    cpdef use_saved_rover_config(self, nrovers):
        cdef list config_input

        config_input = []
        with open('Output_Data/Rover_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        self.rx_init = float(config_input[self.self_id][0])
        self.ry_init = float(config_input[self.self_id][1])
        self.rt_init = float(config_input[self.self_id][2])

    cpdef init_rover_pos_fixed_middle(self, double xd, double yd):  # Set rovers to fixed starting position
        self.rx_init = 0.5*xd
        self.ry_init = 0.5*yd
        self.rt_init = random.uniform(0.0, 360.0)

    cpdef init_rover_pos_random(self, double xd, double yd):  # Randomly set rovers on map
        """
        Rovers given random starting positions and orientations
        :return: rover_positions: np array of size (self.n_rovers, 3)
        """

        self.rx_init = random.uniform(0.0, xd-1.0)
        self.ry_init = random.uniform(0.0, yd-1.0)
        self.rt_init = random.uniform(0.0, 360.0)


    cpdef init_rover_pos_random_concentrated(self, double xd, double yd):
        """
        Rovers given random starting positions within a radius of the center. Starting orientations are random
        :return: rover_positions: np array of size (self.n_rovers, 3)
        """
        cdef double radius, cetner_x, center_y, x, y
        cdef int rov_id

        radius = 4.0; center_x = xd/2.0; center_y = yd/2.0

        x = random.uniform(0.0, xd-1.0)  # Rover X-Coordinate
        y = random.uniform(0.0, yd-1.0)  # Rover Y-Coordinate

        while x > (center_x + radius) or x < (center_x - radius):
            x = random.uniform(0.0, xd-1.0)  # Rover X-Coordinate
        while y > (center_y + radius) or y < (center_y - radius):
            y = random.uniform(0.0, yd-1.0)  # Rover Y-Coordinate

        self.rx_init = x  # Rover X-Coordinate
        self.ry_init = y  # Rover Y-Coordinate
        self.rt_init = random.uniform(0.0, 360.0)  # Rover orientation


    cpdef init_rover_pos_bottom_center(self, double xd, double yd):
        self.rx_init = random.uniform((xd/2.0) - 5.0, (xd/2.0) + 5.0)
        self.ry_init = random.uniform(0.0, 2.0)
        self.rt_init = random.uniform(0.0, 360.0)  # Rover orientation

    cpdef step(self, double [:] joint_action, double wx, double wy):
        """
        :param joint_action: np array containing output from NN. Array size is (nrovers, 2)
        :return: Joint state of rovers (NN inputs), Done, and Global Reward
        """
        cdef double x, y, theta

        joint_action = np.clip(joint_action, -1.0, 1.0)

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
        if 0.0 <= (self.rover_x + x) < (wx-1.0) and 0.0 <= (self.rover_y + y) < (wy-1.0):
            self.rover_x += x
            self.rover_y += y
        self.rover_theta = theta

    cpdef rover_sensor_scan(self, dict rovers, double [:, :] pois, int num_rovers, int num_poi):
        cdef double poi_x, poi_y, poi_value, angle, dist, rov_x, rov_y
        cdef double num_poi_double, num_agents_double
        cdef int rover_id, poi_id, bracket, other_rover_id, num_agents
        cdef double [:] rover_state, poi_state
        cdef list temp_poi_dist_list, temp_rover_dist_list

        rover_state = np.zeros(int(360.0 / self.angle_res))
        poi_state = np.zeros(int(360.0 / self.angle_res))
        temp_poi_dist_list = [[] for _ in range(int(360.0 / self.angle_res))]
        temp_rover_dist_list = [[] for _ in range(int(360.0 / self.angle_res))]

        # Log POI distances into brackets
        for poi_id in range(num_poi):
            poi_x = pois[poi_id, 0]
            poi_y = pois[poi_id, 1]
            poi_value = pois[poi_id, 2]

            angle, dist = self.get_angle_dist(self.rover_x, self.rover_y, poi_x, poi_y)

            angle -= self.rover_theta
            if angle < 0.0:
                angle += 360.0
            elif angle > 360.0:
                angle -= 360.0

            bracket = int(angle/self.angle_res)
            if bracket >= len(temp_poi_dist_list):
                bracket = len(temp_poi_dist_list) - 1
            if dist < 1.0:  # Clip distance to not overwhelm tanh in NN
                dist = 1.0

            temp_poi_dist_list[bracket].append(poi_value/dist)

        # Log rover distances into brackets
        for rover_id in range(num_rovers):
            if self.self_id == rover_id: # Ignore self
                continue
            rov_x = rovers["Rover{0}".format(rover_id)].rover_x
            rov_y = rovers["Rover{0}".format(rover_id)].rover_y

            angle, dist = self.get_angle_dist(self.rover_x, self.rover_y, rov_x, rov_y)

            angle -= self.rover_theta
            if angle < 0.0:
                angle += 360.0
            elif angle > 360.0:
                angle -= 360.0


            if dist < 1.0:  # Clip distance to not overwhelm sigmoid in NN
                dist = 1.0

            bracket = int(angle / self.angle_res)
            if bracket >= len(temp_rover_dist_list):
                bracket = len(temp_rover_dist_list) - 1
            temp_rover_dist_list[bracket].append(1/dist)

        # Encode the information into the state vector
        for bracket in range(int(360/self.angle_res)):
            # POIs
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

            # Rovers
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

    cpdef rover_sensor_scan_spatial(self, dict rovers, double [:, :] pois, int [:] poi_obs, int num_rovers, int num_poi):
        """
        Sensor scan function to be used for tests involving spatial coupling
        :param self: 
        :param rovers: 
        :param pois: 
        :param poi_obs: 
        :param num_rovers: 
        :param num_poi: 
        :return: 
        """
        cdef double poi_x, poi_y, poi_value, angle, dist, rov_x, rov_y
        cdef double num_poi_double, num_agents_double
        cdef int rover_id, poi_id, bracket, other_rover_id, num_agents
        cdef double [:] rover_state, poi_state
        cdef list temp_poi_dist_list, temp_rover_dist_list

        assert(self.sensor_readings.size() == int(360.0 / self.angle_res) + num_poi)

        rover_state = np.zeros(int(360.0 / self.angle_res))
        poi_state = np.zeros(int(360.0 / self.angle_res))
        temp_poi_dist_list = [[] for _ in range(int(360.0 / self.angle_res))]
        temp_rover_dist_list = [[] for _ in range(int(360.0 / self.angle_res))]

        # Log POI distances into brackets
        for poi_id in range(num_poi):
            poi_x = pois[poi_id, 0]
            poi_y = pois[poi_id, 1]
            poi_value = pois[poi_id, 2]

            angle, dist = self.get_angle_dist(self.rover_x, self.rover_y, poi_x, poi_y)

            angle -= self.rover_theta
            if angle < 0.0:
                angle += 360.0
            elif angle > 360.0:
                angle -= 360.0

            bracket = int(angle/self.angle_res)
            if bracket >= len(temp_poi_dist_list):
                bracket = len(temp_poi_dist_list) - 1
            if dist < 1.0:  # Clip distance to not overwhelm tanh in NN
                dist = 1.0

            temp_poi_dist_list[bracket].append(poi_value/dist)
            self.sensor_readings[int(360.0 / self.angle_res) + poi_id] = poi_obs[poi_id]

        # Log rover distances into brackets
        for rover_id in range(num_rovers):
            if self.self_id == rover_id: # Ignore self
                continue
            rov_x = rovers["Rover{0}".format(rover_id)].rover_x
            rov_y = rovers["Rover{0}".format(rover_id)].rover_y

            angle, dist = self.get_angle_dist(self.rover_x, self.rover_y, rov_x, rov_y)

            angle -= self.rover_theta
            if angle < 0.0:
                angle += 360.0
            elif angle > 360.0:
                angle -= 360.0


            if dist < 1.0:  # Clip distance to not overwhelm sigmoid in NN
                dist = 1.0

            bracket = int(angle / self.angle_res)
            if bracket >= len(temp_rover_dist_list):
                bracket = len(temp_rover_dist_list) - 1
            temp_rover_dist_list[bracket].append(1/dist)

        # Encode the information into the state vector
        for bracket in range(int(360/self.angle_res)):
            # POIs
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

            # Rovers
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
        elif angle > 360.0:
            angle -= 360.0
        elif math.isnan(angle):
            angle = 0.0

        dist = math.sqrt((vx**2) + (vy**2))

        return angle, dist