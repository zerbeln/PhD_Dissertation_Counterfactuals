import numpy as np
import math
import random
import os
import csv
from parameters import parameters as p


class RoverDomain:
    def __init__(self, new_world=True):
        # World attributes
        self.world_x = p["x_dim"]
        self.world_y = p["y_dim"]
        self.num_pois = p["n_poi"]
        self.n_rovers = p["n_rovers"]
        self.c_req = p["coupling"]
        self.min_dist = p["min_distance"]
        self.obs_radius = p["observation_radius"]
        self.create_new_world_config = new_world
        self.rover_steps = p["steps"]
        self.stat_runs = p["stat_runs"]

        # Rover Information
        self.rover_path = np.zeros((self.n_rovers, (self.rover_steps + 1), 3))
        self.initial_rover_positions = np.zeros((self.n_rovers, 3))

        # POI Information
        self.pois = np.zeros((self.num_pois, 3))  # [X, Y, Val]
        self.poi_visits = np.zeros((self.num_pois, self.n_rovers))

    def inital_world_setup(self, srun):
        """
        Set POI positions and POI values
        :return: none
        """

        if self.create_new_world_config:
            for sr in range(self.stat_runs):
                self.pois = np.zeros((self.num_pois, 3))
                # Initialize POI positions and values
                self.init_poi_pos_random()
                self.init_poi_vals_identical(10.0)
                self.save_poi_configuration(sr)

                # Initialize Rover Positions
                self.init_rover_pos_random()
                self.save_rover_configuration(sr)
        else:
            # Initialize POI positions and values
            self.pois = np.zeros((self.num_pois, 3))
            self.use_saved_poi_configuration(srun)

        self.rover_path = np.zeros((self.n_rovers, (self.rover_steps + 1), 3))

    def calc_global(self, rovers):
        """
        Calculates global reward for current world state.
        :return: global_reward
        """
        global_reward = 0.0

        for poi_id in range(self.num_pois):
            rover_distances = np.zeros(self.n_rovers)
            observer_count = 0

            for agent_id in range(self.n_rovers):
                # Calculate distance between agent and POI
                x_distance = self.pois[poi_id, 0] - rovers["Rover{0".format(agent_id)].rover_x
                y_distance = self.pois[poi_id, 1] - rovers["Rover{0".format(agent_id)].rover_y
                distance = math.sqrt((x_distance**2) + (y_distance**2))

                if distance < self.min_dist:
                    distance = self.min_dist

                rover_distances[agent_id] = distance

                # Check if agent observes poi and update observer count if true
                if distance < self.obs_radius:
                    observer_count += 1

            # Update global reward if POI is observed
            if observer_count >= self.c_req:
                global_reward += self.pois[poi_id, 2]

        return global_reward

    def clear_rover_path(self):
        """
        Clears the rover path array by setting it to 0
        """
        # self.poi_visits = np.zeros(p["n_poi"])
        self.rover_path = np.zeros((self.n_rovers, (self.rover_steps + 1), 3))

    def clear_poi_visit_list(self):
        """
        Clears the POI visit markers by setting them to 0
        """
        self.poi_visits = np.zeros((self.num_pois, self.n_rovers))

    def update_rover_path(self, rovers, steps):
        """
        Update the array which tracks each rover's positon at each time step
        """
        for rover_id in range(self.n_rovers):
            self.rover_path[rover_id, steps+1, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
            self.rover_path[rover_id, steps+1, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
            self.rover_path[rover_id, steps+1, 2] = rovers["Rover{0}".format(rover_id)].pos[2]

    def determine_poi_visits(self, rovers):
        """
        Check to see which POI are observed successfully
        """

        for poi_id in range(self.num_pois):
            poi_x = self.pois[poi_id, 0]
            poi_y = self.pois[poi_id, 1]
            for rover_id in range(self.n_rovers):
                rov_x = rovers["Rover{0}".format(rover_id)].rover_x
                rov_y = rovers["Rover{0}".format(rover_id)].rover_y
                dist = math.sqrt((rov_x - poi_x)**2 + (rov_y - poi_y)**2)

                if dist < self.obs_radius:
                    self.poi_visits[poi_id, rover_id] = 1

    def save_poi_configuration(self, srun):
        """
        Saves world configuration to a csv file in a folder called Output_Data
        :Output: One CSV file containing POI postions and POI values
        """
        dir_name = 'Output_Data/SRUN{0}'.format(srun)  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        pfile_name = os.path.join(dir_name, 'POI_Config.csv')

        with open(pfile_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for poi_id in range(self.num_pois):
                writer.writerow(self.pois[poi_id, :])

    def use_saved_poi_configuration(self, srun):
        """
        Load in a save POI configuation from a CSV file
        """
        config_input = []
        with open('Output_Data/SRUN{0}/POI_Config.csv'.format(srun)) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        for poi_id in range(self.num_pois):
            self.pois[poi_id, 0] = float(config_input[poi_id][0])
            self.pois[poi_id, 1] = float(config_input[poi_id][1])
            self.pois[poi_id, 2] = float(config_input[poi_id][2])

    def save_rover_configuration(self, srun):
        """
        Saves rover positions to a csv file in a folder called Output_Data
        :Output: CSV file containing rover starting positions
        """
        dir_name = 'Output_Data/SRUN{0}'.format(srun)  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        pfile_name = os.path.join(dir_name, 'Rover_Config.csv')

        row = np.zeros(3)
        with open(pfile_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for rov_id in range(self.n_rovers):
                row[0] = self.initial_rover_positions[rov_id, 0]
                row[1] = self.initial_rover_positions[rov_id, 1]
                row[2] = self.initial_rover_positions[rov_id, 2]
                writer.writerow(row[:])

    # ROVER POSITION FUNCTIONS ---------------------------------------------------------------------------------------
    def init_rover_pos_random(self):  # Randomly set rovers on map
        """
        Rovers given random starting positions and orientations
        :return: rover_positions: np array of size (self.n_rovers, 3)
        """

        for rov_id in range(self.n_rovers):
            rover_x = random.uniform(0.0, self.world_x-1.0)
            rover_y = random.uniform(0.0, self.world_y-1.0)
            rover_theta = random.uniform(0.0, 360.0)

            rover_too_close = True
            while rover_too_close:
                count = 0
                for poi_id in range(self.num_pois):
                    x_dist = self.pois[poi_id, 0] - rover_x
                    y_dist = self.pois[poi_id, 1] - rover_y
                    dist = math.sqrt((x_dist**2) + (y_dist**2))

                    if dist < (self.obs_radius+2):
                        count += 1

                if count == 0:
                    rover_too_close = False
                else:
                    rover_x = random.uniform(0.0, self.world_x - 1.0)
                    rover_y = random.uniform(0.0, self.world_y - 1.0)
                    count = 0

            self.initial_rover_positions[rov_id, 0] = rover_x
            self.initial_rover_positions[rov_id, 1] = rover_y
            self.initial_rover_positions[rov_id, 2] = rover_theta

    def init_rover_pos_random_concentrated(self):
        """
        Rovers given random starting positions within a radius of the center. Starting orientations are random
        :return: rover_positions: np array of size (self.n_rovers, 3)
        """
        radius = 4.0
        center_x = self.world_x/2.0
        center_y = self.world_y/2.0

        for rov_id in range(self.n_rovers):
            x = random.uniform(0.0, self.world_x-1.0)  # Rover X-Coordinate
            y = random.uniform(0.0, self.world_y-1.0)  # Rover Y-Coordinate

            while x > (center_x + radius) or x < (center_x - radius):
                x = random.uniform(0.0, self.world_x-1.0)  # Rover X-Coordinate
            while y > (center_y + radius) or y < (center_y - radius):
                y = random.uniform(0.0, self.world_y-1.0)  # Rover Y-Coordinate

            self.initial_rover_positions[rov_id, 0] = x  # Rover X-Coordinate
            self.initial_rover_positions[rov_id, 1] = y  # Rover Y-Coordinate
            self.initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)  # Rover orientation

    def init_rover_pos_fixed_middle(self):  # Set rovers to fixed starting position
        """
        Create a starting position for the rover near the center of the world
        """
        for rov_id in range(self.n_rovers):
            self.initial_rover_positions[rov_id, 0] = 0.5*self.world_x
            self.initial_rover_positions[rov_id, 1] = 0.5*self.world_y
            self.initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)

    # POI POSITION FUNCTIONS ------------------------------------------------------------------------------------------
    def init_poi_pos_random(self):  # Randomly set POI on the map
        """
        POI positions set randomly across the map (but not in range of any rover)
        :return: self.pois: np array of size (npoi, 2)
        """
        for poi_id in range(self.num_pois):
            x = random.uniform(0, self.world_x-1.0)
            y = random.uniform(0, self.world_y-1.0)

            poi_too_close = True
            while poi_too_close:
                count = 0
                for p_id in range(self.num_pois):
                    if p_id != poi_id:
                        x_dist = x - self.pois[p_id, 0]
                        y_dist = y - self.pois[p_id, 1]

                        dist = math.sqrt((x_dist**2) + (y_dist**2))
                        if dist < (self.obs_radius + 2):
                            count += 1

                if count == 0:
                    poi_too_close = False
                else:
                    x = random.uniform(0, self.world_x - 1.0)
                    y = random.uniform(0, self.world_y - 1.0)
                    count = 0

            self.pois[poi_id, 0] = x
            self.pois[poi_id, 1] = y

    def init_poi_pos_circle(self):
        """
            POI positions are set in a circle around the center of the map at a specified radius.
            :return: self.pois: np array of size (npoi, 2)
        """
        radius = 15.0
        interval = float(360/self.num_pois)

        x = self.world_x/2.0
        y = self.world_y/2.0
        theta = 0.0

        for poi_id in range(self.num_pois):
            self.pois[poi_id, 0] = x + radius*math.cos(theta*math.pi/180)
            self.pois[poi_id, 1] = y + radius*math.sin(theta*math.pi/180)
            theta += interval

    def init_poi_pos_concentric_circles(self):
        """
        POI positions are set in a circle around the center of the map at a specified radius.
        :return: self.pois: np array of size (npoi, 2)
        """
        assert(self.num_pois == 12)
        inner_radius = 6.5
        outter_radius = 15.0
        interval = float(360 /(self.num_pois/2))

        x = self.world_x/2.0
        y = self.world_y/2.0
        inner_theta = 0.0
        outter_theta = 0.0

        for poi_id in range(self.num_pois):
            if poi_id < 6:
                self.pois[poi_id, 0] = x + inner_radius * math.cos(inner_theta * math.pi / 180)
                self.pois[poi_id, 1] = y + inner_radius * math.sin(inner_theta * math.pi / 180)
                inner_theta += interval
            else:
                self.pois[poi_id, 0] = x + outter_radius * math.cos(outter_theta * math.pi / 180)
                self.pois[poi_id, 1] = y + outter_radius * math.sin(outter_theta * math.pi / 180)
                outter_theta += interval


    def init_poi_pos_two_poi(self):
        """
        Sets two POI on the map, one on the left, one on the right at Y-Dimension/2
        :return: self.pois: np array of size (npoi, 2)
        """
        assert(self.num_pois == 2)

        self.pois[0, 0] = 1.0; self.pois[0, 1] = self.world_y/2.0
        self.pois[1, 0] = (self.world_x-2.0); self.pois[1, 1] = self.world_y/2.0


    def init_poi_pos_four_corners(self):  # Statically set 4 POI (one in each corner)
        """
        Sets 4 POI on the map in a box formation around the center
        :return: self.pois: np array of size (npoi, 2)
        """
        assert(self.num_pois == 4)  # There must only be 4 POI for this initialization

        self.pois[0, 0] = 2.0; self.pois[0, 1] = 2.0  # Bottom left
        self.pois[1, 0] = 2.0; self.pois[1, 1] = (self.world_y - 2.0)  # Top left
        self.pois[2, 0] = (self.world_x - 2.0); self.pois[2, 1] = 2.0  # Bottom right
        self.pois[3, 0] = (self.world_x - 2.0); self.pois[3, 1] = (self.world_y - 2.0)  # Top right

    # POI VALUE FUNCTIONS -----------------------------------------------------------------------------------
    def init_poi_vals_random(self):
        """
        POI values randomly assigned 1-10
        :return: poi_vals: array of size(npoi)
        """

        for poi_id in range(self.num_pois):
            self.pois[poi_id, 2] = float(random.randint(1, 12))

    def init_poi_vals_identical(self, poi_val):
        """
        POI values set to fixed, identical value
        :return: poi_vals: array of size(npoi)
        """

        for poi_id in range(self.num_pois):
            self.pois[poi_id, 2] = poi_val

