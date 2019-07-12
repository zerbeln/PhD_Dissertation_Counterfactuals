import sys
import math
from AADI_RoverDomain.rover_setup import *
from AADI_RoverDomain.parameters import Parameters as p


class RoverDomain:

    def __init__(self):
        self.num_agents = p.num_rovers
        self.obs_radius = p.min_observation_dist

        #Gym compatible attributes
        self.observation_space = np.zeros((1, int(2*360 / p.angle_resolution)))
        self.istep = 0  # Current Step counter

        # Initialize POI containers tha track POI position
        self.poi_pos = init_poi_positions_random()
        self.poi_values = init_poi_values_random()

        # Initialize rover position container
        self.rover_pos = init_rover_positions_random()
        self.rover_initial_pos = self.rover_pos.copy()  # Track initial setup

        #Rover path trace for trajectory-wide global reward computation and vizualization purposes
        self.rover_path = np.zeros(((p.num_steps + 1), self.num_agents, 3))

    def reset_world(self):
        """
        Changes rovers' starting positions and POI positions and values according to specified functions
        :return: none
        """
        self.rover_pos = init_rover_positions_random()
        self.rover_initial_pos = self.rover_pos.copy()  # Track initial setup
        self.poi_pos = init_poi_positions_random()
        self.poi_values = init_poi_values_random()
        self.rover_path = np.zeros(((p.num_steps + 1), self.num_agents, 3))
        self.istep = 0

        for rover_id in range(self.num_agents):  # Record intial positions
            self.rover_path[self.istep, rover_id, 0] = self.rover_pos[rover_id, 0]
            self.rover_path[self.istep, rover_id, 1] = self.rover_pos[rover_id, 1]
            self.rover_path[self.istep, rover_id, 2] = self.rover_pos[rover_id, 2]

    def reset_to_init(self):
        """
        Resets rovers to starting positions (does not alter the starting positions)
        :return: none
        """
        self.rover_pos = self.rover_initial_pos.copy()
        self.rover_path = np.zeros(((p.num_steps + 1), self.num_agents, 3))
        self.istep = 0

        for rover_id in range(self.num_agents):  # Record initial positions
            self.rover_path[self.istep, rover_id, 0] = self.rover_pos[rover_id, 0]
            self.rover_path[self.istep, rover_id, 1] = self.rover_pos[rover_id, 1]
            self.rover_path[self.istep, rover_id, 2] = self.rover_pos[rover_id, 2]

    def step(self, joint_action):
        """
        :param joint_action: np array containing output from NN. Array size is (nrovers, 2)
        :return: Joint state of rovers (NN inputs), Done, and Global Reward
        """
        self.istep += 1
        joint_action = joint_action.clip(-1.0, 1.0)

        # Update rover positions
        for rover_id in range(self.num_agents):
            magnitude = 0.5 * (joint_action[rover_id, 0] + 1)  # [-1,1] --> [0,1]

            joint_action[rover_id, 1] /= 2.0  # Theta (bearing constrained to be within 90 degree turn from heading)
            theta = joint_action[rover_id, 1] * 180 + self.rover_pos[rover_id, 2]
            if theta > 360: theta -= 360
            if theta < 0: theta += 360
            self.rover_pos[rover_id, 2] = theta

            # Update position
            x = magnitude * math.cos(math.radians(theta))
            y = magnitude * math.sin(math.radians(theta))

            self.rover_pos[rover_id, 0] += x
            self.rover_pos[rover_id, 1] += y


        # Update rover path
        for rover_id in range(self.num_agents):
            self.rover_path[self.istep, rover_id, 0] = self.rover_pos[rover_id, 0]
            self.rover_path[self.istep, rover_id, 1] = self.rover_pos[rover_id, 1]
            self.rover_path[self.istep, rover_id, 2] = self.rover_pos[rover_id, 2]

        # Computes done
        done = int(self.istep >= p.num_steps)

        joint_state = self.get_joint_state()

        g_reward = self.calc_global()

        return joint_state, done, g_reward

    def get_joint_state(self):
        """
        joint_state is an array of size [nrovers][8] containing inputs for NN
        :return: joint_state
        """
        joint_state = []

        for rover_id in range(self.num_agents):
            self_x = self.rover_pos[rover_id, 0]; self_y = self.rover_pos[rover_id, 1]; self_orient = self.rover_pos[rover_id, 2]

            rover_state = [0.0 for _ in range(int(360 / p.angle_resolution))]
            poi_state = [0.0 for _ in range(int(360 / p.angle_resolution))]
            temp_poi_dist_list = [[] for _ in range(int(360 / p.angle_resolution))]
            temp_rover_dist_list = [[] for _ in range(int(360 / p.angle_resolution))]

            # Log POI distances into brackets
            for loc, value in zip(self.poi_pos, self.poi_values):

                angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])

                if dist >= self.obs_radius: continue  # Observability radius

                angle -= self_orient
                if angle < 0: angle += 360

                bracket = int(angle / p.angle_resolution)
                if bracket >= len(temp_poi_dist_list):
                    print("ERROR: BRACKET EXCEED LIST", bracket, len(temp_poi_dist_list))
                    bracket = len(temp_poi_dist_list) - 1
                if dist < p.min_distance:  # Clip distance to not overwhelm tanh in NN
                    dist = p.min_distance

                temp_poi_dist_list[bracket].append(value/dist)

            # Log rover distances into brackets
            for id, loc in enumerate(self.rover_pos):
                if id == rover_id: continue  # Ignore self

                angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])

                if dist >= self.obs_radius: continue  # Observability radius

                angle -= self_orient
                if angle < 0: angle += 360

                if dist < p.min_distance:  # Clip distance to not overwhelm sigmoid in NN
                    dist = p.min_distance

                bracket = int(angle / p.angle_resolution)
                if bracket >= len(temp_rover_dist_list):
                    print("ERROR: BRACKET EXCEED LIST", bracket, len(temp_rover_dist_list))
                    bracket = len(temp_rover_dist_list) - 1
                temp_rover_dist_list[bracket].append(1/dist)


            # Encode the information into the state vector
            for bracket in range(int(360 / p.angle_resolution)):
                # POIs
                num_poi = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
                if num_poi > 0:
                    if p.sensor_model == 'density':
                        poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi  # Density Sensor
                    elif p.sensor_model == 'closest':
                        poi_state[bracket] = max(temp_poi_dist_list[bracket])  # Closest Sensor
                    else:
                        sys.exit('Incorrect sensor model')
                else:
                    poi_state[bracket] = -1.0

                # Rovers
                num_agents = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
                if num_agents > 0:
                    if p.sensor_model == 'density':
                        rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_agents  # Density Sensor
                    elif p.sensor_model == 'closest':
                        rover_state[bracket] = max(temp_rover_dist_list[bracket])  # Closest Sensor
                    else:
                        sys.exit('Incorrect sensor model')
                else:
                    rover_state[bracket] = -1.0

            state = rover_state + poi_state  # Append rover and poi to form the full state

            joint_state.append(state)

        return joint_state


    def get_angle_dist(self, x1, y1, x2, y2):
        """
        Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        :param x1: X-Position of rover
        :param y1: Y-Position of rover
        :param x2: X-Position of POI or other rover
        :param y2: Y-Position of POI or other rover
        :return: angle, dist
        """
        v1 = x2 - x1; v2 = y2 - y1
        angle = np.rad2deg(np.arctan2(v1, v2))
        if angle < 0:
            angle += 360

        if math.isnan(angle):
            angle = 0.0

        dist = (v1 * v1) + (v2 * v2)
        dist = math.sqrt(dist)

        return angle, dist


    def calc_global(self):
        """
        Calculates global reward for current world state.
        :return: global_reward
        """
        number_agents = p.num_rovers
        number_pois = p.num_pois
        inf = 1000.00
        global_reward = 0.0

        for poi_id in range(number_pois):
            rover_distances = [0.0 for i in range(number_agents)]
            observer_count = 0
            summed_observer_distances = 0.0

            for agent_id in range(number_agents):
                # Calculate distance between agent and POI
                x_distance = self.poi_pos[poi_id, 0] - self.rover_pos[agent_id, 0]
                y_distance = self.poi_pos[poi_id, 1] - self.rover_pos[agent_id, 1]
                distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                if distance < p.min_distance:
                    distance = p.min_distance

                rover_distances[agent_id] = distance

                # Check if agent observes poi and update observer count if true
                if distance < self.obs_radius:
                    observer_count += 1

            # Update global reward if POI is observed
            if observer_count >= p.coupling:
                for observer in range(p.coupling):
                    summed_observer_distances += min(rover_distances)
                    od_index = rover_distances.index(min(rover_distances))
                    rover_distances[od_index] = inf

                global_reward += self.poi_values[poi_id] / ((1 / p.coupling) * summed_observer_distances)

        return global_reward
