import random, sys
import numpy as np
import math
from heterogeneous_rovers import *
from parameters import Parameters as p


class RoverDomain:

    def __init__(self):
        self.num_agents = p.num_rovers * p.num_types
        self.obs_radius = p.activation_dist

        #Gym compatible attributes
        self.observation_space = np.zeros((1, int(2*360 / p.angle_resolution)))
        self.istep = 0 #Current Step counter

        # Initialize POI containers tha track POI position
        self.poi_pos = init_poi_positions_four_corners()
        self.poi_value = init_poi_values_fixed()

        # Initialize rover position container
        self.rover_pos = init_rover_positions_fixed()
        self.rover_initial_pos = self.rover_pos.copy()  # Track initial setup

        #Rover path trace for trajectory-wide global reward computation and vizualization purposes
        self.rover_path = np.zeros(((p.num_steps + 1), self.num_agents, 3))

    def reset(self):  # Resets entire world (For new stat run)
        self.rover_pos = init_rover_positions_fixed()
        self.rover_initial_pos = self.rover_pos.copy()  # Track initial setup
        self.poi_pos = init_poi_positions_four_corners()
        self.poi_value = init_poi_values_fixed()
        self.rover_path = np.zeros(((p.num_steps + 1), self.num_agents, 3))
        self.istep = 0

        for rover_id in range(self.num_agents):  # Record intial positions
            self.rover_path[self.istep, rover_id, 0] = self.rover_pos[rover_id, 0]
            self.rover_path[self.istep, rover_id, 1] = self.rover_pos[rover_id, 1]
            self.rover_path[self.istep, rover_id, 2] = self.rover_pos[rover_id, 2]  # Tracks rover type

    def reset_to_init(self):
        self.rover_pos = self.rover_initial_pos.copy()
        self.rover_path = np.zeros(((p.num_steps + 1), self.num_agents, 3))
        self.istep = 0

        for rover_id in range(self.num_agents):  # Record initial positions
            self.rover_path[self.istep, rover_id, 0] = self.rover_pos[rover_id, 0]
            self.rover_path[self.istep, rover_id, 1] = self.rover_pos[rover_id, 1]
            self.rover_path[self.istep, rover_id, 2] = self.rover_pos[rover_id, 2]  # Tracks rover type

    def step(self, joint_action):
        self.istep += 1

        for rover_id in range(self.num_agents):
            self.rover_pos[rover_id, 0] += joint_action[rover_id, 0]
            self.rover_pos[rover_id, 1] += joint_action[rover_id, 1]


        #Append rover path
        for rover_id in range(self.num_agents):
            self.rover_path[self.istep, rover_id, 0] = self.rover_pos[rover_id, 0]
            self.rover_path[self.istep, rover_id, 1] = self.rover_pos[rover_id, 1]
            self.rover_path[self.istep, rover_id, 2] = self.rover_pos[rover_id, 2]  # Tracks rover type

        #Compute done
        done = int(self.istep >= p.num_steps)

        if p.team_types == "heterogeneous":
            joint_state = self.get_hetero_joint_state()
        else:
            joint_state = self.get_joint_state()

        return joint_state, done

    def get_joint_state(self):
        joint_state = []

        for rover_id in range(self.num_agents):
            self_x = self.rover_pos[rover_id, 0]; self_y = self.rover_pos[rover_id, 1]

            rover_state = [0.0 for _ in range(int(360 / p.angle_resolution))]
            poi_state = [0.0 for _ in range(int(360 / p.angle_resolution))]
            temp_poi_dist_list = [[] for _ in range(int(360 / p.angle_resolution))]
            temp_rover_dist_list = [[] for _ in range(int(360 / p.angle_resolution))]

            # Log all distance into brackets for POIs
            for loc, value in zip(self.poi_pos, self.poi_value):

                angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
                if dist >= self.obs_radius:
                    continue  # Observability radius

                bracket = int(angle / p.angle_resolution)
                if dist < p.min_distance:  # Clip distance to not overwhelm tanh in NN
                    dist = p.min_distance
                temp_poi_dist_list[bracket].append((value/dist))

            # Log all distance into brackets for other drones
            for id, loc in enumerate(self.rover_pos):
                if id == rover_id:
                    continue  # Ignore self

                angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
                if dist >= self.obs_radius:
                    continue  # Observability radius

                if dist < p.min_distance:  # Clip distance to not overwhelm tanh in NN
                    dist = p.min_distance
                bracket = int(angle / p.angle_resolution)
                temp_rover_dist_list[bracket].append((1/dist))


            ####Encode the information onto the state
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

    def get_hetero_joint_state(self):  # Two rover types with different sensors
        joint_state = []

        assert(p.num_types == 2)

        for rover_id in range(self.num_agents):
            self_x = self.rover_pos[rover_id, 0]; self_y = self.rover_pos[rover_id, 1]
            rover_type = self.rover_pos[rover_id, 2]

            rover_state = [0.0 for _ in range(int(360 / p.angle_resolution))]
            poi_state = [0.0 for _ in range(int(360 / p.angle_resolution))]
            temp_poi_dist_list = [[] for _ in range(int(360 / p.angle_resolution))]
            temp_rover_dist_list = [[] for _ in range(int(360 / p.angle_resolution))]

            # Log all distance into brackets for POIs
            for loc, value in zip(self.poi_pos, self.poi_value):

                angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
                if dist >= self.obs_radius:
                    continue  # Observability radius

                bracket = int(angle / p.angle_resolution)
                if dist < p.min_distance:  # Clip distance to not overwhelm tanh in NN
                    dist = p.min_distance
                temp_poi_dist_list[bracket].append((value/dist))

            # Log all distance into brackets for other drones
            for id, loc in enumerate(self.rover_pos):
                if id == rover_id:
                    continue  # Ignore self

                angle, dist = self.get_angle_dist(self_x, self_y, loc[0], loc[1])
                if dist >= self.obs_radius:
                    continue  # Observability radius

                if dist < p.min_distance:  # Clip distance to not overwhelm tanh in NN
                    dist = p.min_distance
                bracket = int(angle / p.angle_resolution)
                temp_rover_dist_list[bracket].append((1/dist))


            ####Encode the information onto the state
            for bracket in range(int(360 / p.angle_resolution)):
                # POIs
                num_poi = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
                if num_poi > 0:
                    if rover_type == 1:
                        poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi  # Density Sensor
                    elif rover_type == 0:
                        poi_state[bracket] = max(temp_poi_dist_list[bracket])  # Closest Sensor
                    else:
                        sys.exit('Incorrect sensor model')
                else:
                    poi_state[bracket] = 0.0

                # Rovers
                num_agents = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
                if num_agents > 0:
                    if rover_type == 1:
                        rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_agents  # Density Sensor
                    elif rover_type == 0:
                        rover_state[bracket] = max(temp_rover_dist_list[bracket])  # Closest Sensor
                    else:
                        sys.exit('Incorrect sensor model')
                else:
                    rover_state[bracket] = 0.0

            state = rover_state + poi_state  # Append rover and poi to form the full state

            joint_state.append(state)

        return joint_state

    def get_angle_dist(self, x1, y1, x2, y2):  # Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        v1 = x2 - x1; v2 = y2 - y1
        angle = np.rad2deg(np.arctan2(v1, v2))
        if angle < 0:
            angle += 360

        dist = (v1 * v1) + (v2 * v2)
        dist = math.sqrt(dist)

        return angle, dist


    # def reset_poi_pos(self):
    #
    #     if self.args.unit_test == 1: #Unit_test
    #         self.poi_pos[0] = [0,1]
    #         return
    #
    #     if self.args.unit_test == 2: #Unit_test
    #         if random.random() < 0.5:
    #             self.poi_pos[0] = [4,0]
    #         else:
    #             self.poi_pos[0] = [4,9]
    #         return
    #
    #     start = 0.0; end = self.args.dim_x - 1.0
    #     rad = int(self.args.dim_x / math.sqrt(3) / 2.0)
    #     center = int((start + end) / 2.0)
    #
    #     if self.args.poi_rand: #Random
    #         for i in range(self.args.num_poi):
    #             if i % 3 == 0:
    #                 x = randint(start, center - rad - 1)
    #                 y = randint(start, end)
    #             elif i % 3 == 1:
    #                 x = randint(center + rad + 1, end)
    #                 y = randint(start, end)
    #             elif i % 3 == 2:
    #                 x = randint(center - rad, center + rad)
    #                 if random.random()<0.5:
    #                     y = randint(start, center - rad - 1)
    #                 else:
    #                     y = randint(center + rad + 1, end)
    #
    #             self.poi_pos[i] = [x, y]
    #
    #     else: #Not_random
    #         for i in range(self.args.num_poi):
    #             if i % 4 == 0:
    #                 x = start + int(i/4) #randint(start, center - rad - 1)
    #                 y = start + int(i/3)
    #             elif i % 4 == 1:
    #                 x = end - int(i/4) #randint(center + rad + 1, end)
    #                 y = start + int(i/4)#randint(start, end)
    #             elif i % 4 == 2:
    #                 x = start+ int(i/4) #randint(center - rad, center + rad)
    #                 y = end - int(i/4) #randint(start, center - rad - 1)
    #             else:
    #                 x = end - int(i/4) #randint(center - rad, center + rad)
    #                 y = end - int(i/4) #randint(center + rad + 1, end)
    #             self.poi_pos[i] = [x, y]


    # def reset_rover_pos(self):
    #     start = 1.0; end = self.args.dim_x - 1.0
    #     rad = int(self.args.dim_x / math.sqrt(3) / 2.0)
    #     center = int((start + end) / 2.0)
    #
    #     if self.args.unit_test == 1: #Unit test
    #         self.rover_pos[0] = [end, 0]
    #         return
    #
    #     for rover_id in range(self.args.num_agents):
    #             quadrant = rover_id % 4
    #             if quadrant == 0:
    #                 x = center - 1 - (rover_id / 4) % (center - rad)
    #                 y = center - (rover_id / (4 * center - rad)) % (center - rad)
    #             if quadrant == 1:
    #                 x = center + (rover_id / (4 * center - rad)) % (center - rad)-1
    #                 y = center - 1 + (rover_id / 4) % (center - rad)
    #             if quadrant == 2:
    #                 x = center + 1 + (rover_id / 4) % (center - rad)
    #                 y = center + (rover_id / (4 * center - rad)) % (center - rad)
    #             if quadrant == 3:
    #                 x = center - (rover_id / (4 * center - rad)) % (center - rad)
    #                 y = center + 1 - (rover_id / 4) % (center - rad)
    #             self.rover_pos[rover_id] = [x, y]

    # def render(self):
    #     # Visualize
    #     grid = [['-' for _ in range(self.args.dim_x)] for _ in range(self.args.dim_y)]
    #
    #
    #     # Draw in rover path
    #     for rover_id, path in enumerate(self.rover_path):
    #         for loc in path:
    #             x = int(loc[0]); y = int(loc[1])
    #             if x < self.args.dim_x and y < self.args.dim_y and x >=0 and y >=0:
    #                 grid[x][y] = str(rover_id)
    #
    #     # Draw in food
    #     for poi_pos, poi_status in zip(self.poi_pos, self.poi_status):
    #         x = int(poi_pos[0]);
    #         y = int(poi_pos[1])
    #         marker = '$' if poi_status else '#'
    #         grid[x][y] = marker
    #
    #     for row in grid:
    #         print(row)
    #     print()
    #
    #     print('------------------------------------------------------------------------')
