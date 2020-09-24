import numpy as np
import math
import random
import os
import csv
from parameters import parameters as p


class RoverDomain:
    def __init__(self):
        # World attributes
        self.world_x = p["x_dim"]
        self.world_y = p["y_dim"]
        self.num_pois = p["n_poi"]
        self.n_rovers = p["n_rovers"]
        self.c_req = p["coupling"]
        self.min_dist = p["min_distance"]
        self.obs_radius = p["obs_rad"]
        self.create_new_world_config = p["new_world_config"]
        self.rover_steps = p["n_steps"]

        # Rover Information
        self.rover_path = np.zeros((self.n_rovers, (p["n_steps"] + 1), 3))
        self.final_rover_path = np.zeros((self.n_rovers, p["stat_runs"], (p["n_steps"] + 1), 3))
        self.initial_rover_positions = np.zeros((self.n_rovers, 3))

        # POI Information
        self.pois = np.zeros((self.num_pois, 3))  # [X, Y, Val]
        self.poi_visits = np.zeros((p["n_poi"], p["n_rovers"]))

    def inital_world_setup(self):
        """
        Set POI positions and POI values
        :return: none
        """
        self.pois = np.zeros((self.num_pois, 3))
        if self.create_new_world_config == 1:
            # Initialize Rover positions
            self.init_rover_pos_random_concentrated()
            self.save_rover_configuration()

            # Initialize POI positions and values
            self.init_poi_pos_two_poi()
            self.init_poi_vals_random()
            self.save_poi_configuration()
        else:
            # Initialize POI positions and values
            self.use_saved_poi_configuration()

        self.rover_path = np.zeros((self.n_rovers, (p["n_steps"] + 1), 3))
        self.final_rover_path = np.zeros((self.n_rovers, p["stat_runs"], (p["n_steps"] + 1), 3))

    def clear_rover_path(self):
        self.poi_visits = np.zeros((p["n_poi"], p["n_rovers"]))
        self.rover_path = np.zeros((self.n_rovers, (p["n_steps"] + 1), 3))

    def update_rover_path(self, rovers, steps):
        for rover_id in range(self.n_rovers):
            self.rover_path[rover_id, steps+1, 0] = rovers["Rover{0}".format(rover_id)].rover_x
            self.rover_path[rover_id, steps+1, 1] = rovers["Rover{0}".format(rover_id)].rover_y
            self.rover_path[rover_id, steps+1, 2] = rovers["Rover{0}".format(rover_id)].rover_theta

            # self.determine_poi_visits(rovers, rover_id)

    def determine_poi_visits(self, rovers, rover_id):
        rov_x = rovers["Rover{0}".format(rover_id)].rover_x
        rov_y = rovers["Rover{0}".format(rover_id)].rover_y
        for poi_id in range(self.num_pois):
            poi_x = self.pois[poi_id, 0]
            poi_y = self.pois[poi_id, 1]
            dist = math.sqrt((rov_x - poi_x)**2 + (rov_y - poi_y)**2)

            if dist < self.obs_radius:
                self.poi_visits[poi_id, rover_id] = 1

    def update_final_rover_path(self, srun, rovers, steps):
        for rover_id in range(self.n_rovers):
            self.final_rover_path[rover_id, srun, steps+1, 0] = rovers["Rover{0}".format(rover_id)].rover_x
            self.final_rover_path[rover_id, srun, steps+1, 1] = rovers["Rover{0}".format(rover_id)].rover_y
            self.final_rover_path[rover_id, srun, steps+1, 2] = rovers["Rover{0}".format(rover_id)].rover_theta

    def save_poi_configuration(self):
        """
        Saves world configuration to a csv file in a folder called Output_Data
        :Output: One CSV file containing POI postions and POI values
        """
        dir_name = 'Output_Data/'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        pfile_name = os.path.join(dir_name, 'POI_Config.csv')

        with open(pfile_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for poi_id in range(self.num_pois):
                writer.writerow(self.pois[poi_id, :])

    def use_saved_poi_configuration(self):
        config_input = []
        with open('Output_Data/POI_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        for poi_id in range(self.num_pois):
            self.pois[poi_id, 0] = float(config_input[poi_id][0])
            self.pois[poi_id, 1] = float(config_input[poi_id][1])
            self.pois[poi_id, 2] = float(config_input[poi_id][2])

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

    # ROVER POSITION FUNCTIONS ---------------------------------------------------------------------------------------
    def save_rover_configuration(self):
        """
        Saves rover positions to a csv file in a folder called Output_Data
        :Output: CSV file containing rover starting positions
        """
        dir_name = 'Output_Data/'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        pfile_name = os.path.join(dir_name, 'Rover_Config.csv')

        row = np.zeros(3)
        with open(pfile_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for rov_id in range(p["n_rovers"]):
                row[0] = self.initial_rover_positions[rov_id, 0]
                row[1] = self.initial_rover_positions[rov_id, 1]
                row[2] = self.initial_rover_positions[rov_id, 2]
                writer.writerow(row[:])

    def init_rover_pos_fixed_middle(self):  # Set rovers to fixed starting position
        """
        Create a starting position for the rover near the center of the world
        """
        for rov_id in range(self.n_rovers):
            self.initial_rover_positions[rov_id, 0] = 0.5*p["x_dim"]
            self.initial_rover_positions[rov_id, 1] = 0.5*p["y_dim"]
            self.initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)

    def init_rover_pos_random(self):  # Randomly set rovers on map
        """
        Rovers given random starting positions and orientations
        :return: rover_positions: np array of size (self.n_rovers, 3)
        """

        for rov_id in range(self.n_rovers):
            self.initial_rover_positions[rov_id, 0] = random.uniform(0.0, p["x_dim"]-1.0)
            self.initial_rover_positions[rov_id, 1] = random.uniform(0.0, p["y_dim"]-1.0)
            self.initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)

    def init_rover_pos_random_concentrated(self):
        """
        Rovers given random starting positions within a radius of the center. Starting orientations are random
        :return: rover_positions: np array of size (self.n_rovers, 3)
        """
        radius = 4.0
        center_x = p["x_dim"]/2.0
        center_y = p["y_dim"]/2.0

        for rov_id in range(self.n_rovers):
            x = random.uniform(0.0, p["x_dim"]-1.0)  # Rover X-Coordinate
            y = random.uniform(0.0, p["y_dim"]-1.0)  # Rover Y-Coordinate

            while x > (center_x + radius) or x < (center_x - radius):
                x = random.uniform(0.0, p["x_dim"]-1.0)  # Rover X-Coordinate
            while y > (center_y + radius) or y < (center_y - radius):
                y = random.uniform(0.0, p["y_dim"]-1.0)  # Rover Y-Coordinate

            self.initial_rover_positions[rov_id, 0] = x  # Rover X-Coordinate
            self.initial_rover_positions[rov_id, 1] = y  # Rover Y-Coordinate
            self.initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)  # Rover orientation

    def init_rover_pos_bottom_center(self):
        """
        Set initial rover positions in the bottom, right corner of the world
        """
        for rov_id in range(self.n_rovers):
            self.initial_rover_positions[rov_id, 0] = random.uniform((p["x_dim"] / 2.0) - 5.0, (p["x_dim"] / 2.0) + 5.0)
            self.initial_rover_positions[rov_id, 1] = random.uniform(0.0, 2.0)
            self.initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)  # Rover orientation

    # POI POSITION FUNCTIONS ------------------------------------------------------------------------------------------
    def init_poi_pos_random(self, rovers):  # Randomly set POI on the map
        """
        POI positions set randomly across the map (but not in range of any rover)
        :return: self.pois: np array of size (npoi, 2)
        """
        for poi_id in range(self.num_pois):
            x = random.uniform(0, self.world_x-1.0)
            y = random.uniform(0, self.world_y-1.0)

            rover_id = 0
            while rover_id < self.n_rovers:
                rovx = rovers["Rover{0}".format(rover_id)].rover_x
                rovy = rovers["Rover{0}".format(rover_id)].rover_y
                xdist = x - rovx; ydist = y - rovy
                distance = math.sqrt((xdist**2) + (ydist**2))

                while distance < self.obs_radius:
                    x = random.uniform(0, self.world_x - 1.0)
                    y = random.uniform(0, self.world_y - 1.0)
                    rovx = rovers["Rover{0}".format(rover_id)].rover_x
                    rovy = rovers["Rover{0}".format(rover_id)].rover_y
                    xdist = x - rovx; ydist = y - rovy
                    distance = math.sqrt((xdist ** 2) + (ydist ** 2))
                    rover_id = -1

                rover_id += 1

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


    def init_poi_pos_clusters(self):

        # Low Values Pois
        self.pois[0, 0] = 5.0; self.pois[0, 1] = 15.0
        self.pois[1, 0] = 7.0 ; self.pois[1, 1] = 17.0
        self.pois[2, 0] = 6.0; self.pois[2, 1] = 11.0

        # High Value POIs
        self.pois[3, 0] = 20.0; self.pois[3, 1] = 35.0
        self.pois[4, 0] = 35.0; self.pois[4, 1] = 25.0


    def init_poi_pos_twelve_grid(self):

        assert(self.num_pois == 12)
        poi_id = 0
        for i in range(4):
            for j in range(3):
                self.pois[poi_id, 0] = j * ((self.world_x - 10.0)/2.0)
                self.pois[poi_id, 1] = i * (self.world_y/3.0)

                poi_id += 1


    def init_poi_pos_concentric_squares(self):

        assert(self.num_pois == 8)

        # Inner-Bottom POI
        self.pois[0, 0] = (self.world_x / 2.0)
        self.pois[0, 1] = (self.world_y / 2.0) - 10.0

        # Inner-Right POI
        self.pois[1, 0] = (self.world_x / 2.0) + 10.0
        self.pois[1, 1] = (self.world_y / 2.0)

        # Inner-Top POI
        self.pois[2, 0] = (self.world_x / 2.0)
        self.pois[2, 1] = (self.world_y / 2.0) + 10.0

        # Inner-Left POI
        self.pois[3, 0] = (self.world_x / 2.0) - 10.0
        self.pois[3, 1] = (self.world_y / 2.0)

        # Outter-Bottom-Left POI
        self.pois[4, 0] = (self.world_x / 2.0) - 15.0
        self.pois[4, 1] = (self.world_y / 2.0) - 15

        # Outter-Bottom-Right POI
        self.pois[5, 0] = (self.world_x / 2.0) + 15.0
        self.pois[5, 1] = (self.world_y / 2.0) - 15.0

        # Outter-Top-Left POI
        self.pois[6, 0] = (self.world_x / 2.0) - 15.0
        self.pois[6, 1] = (self.world_y / 2.0) + 15.0

        # Outter-Top-Right POI
        self.pois[7, 0] = (self.world_x / 2.0) + 15.0
        self.pois[7, 1] = (self.world_y / 2.0) + 15.0

    # POI VALUE FUNCTIONS -----------------------------------------------------------------------------------
    def init_poi_vals_random(self):
        """
        POI values randomly assigned 1-10
        :return: poi_vals: array of size(npoi)
        """

        for poi_id in range(self.num_pois):
            self.pois[poi_id, 2] = float(random.randint(1, 12))


    def init_poi_vals_fixed_ascending(self):
        """
        POI values set to fixed, ascending values based on POI ID
        :return: poi_vals: array of size(npoi)
        """

        for poi_id in range(self.num_pois):
            self.pois[poi_id, 2] = poi_id + 1

    def init_poi_vals_fixed_identical(self):
        """
            POI values set to fixed, identical value
            :return: poi_vals: array of size(npoi)
        """

        for poi_id in range(self.num_pois):
            self.pois[poi_id, 2] = 10.0

    def init_poi_vals_half_and_half(self):
        """
        POI values set to fixed value
        :return: poi_vals: array of size(npoi)
        """

        for poi_id in range(self.num_pois):
            if poi_id%2 == 0:
                self.pois[poi_id, 2] = 10.0
            else:
                self.pois[poi_id, 2] = 1.0

    def init_poi_vals_concentric_squares(self):

        assert(self.num_pois == 8)

        # Inner POI Values
        self.pois[0, 2] = 2.0
        self.pois[1, 2] = 2.0
        self.pois[2, 2] = 2.0
        self.pois[3, 2] = 2.0

        # Outter POI Values
        self.pois[4, 2] = 10.0
        self.pois[5, 2] = 10.0
        self.pois[6, 2] = 10.0
        self.pois[7, 2] = 10.0

    def init_poi_vals_concentric_circles(self):
        assert(self.num_pois == 12)

        for poi_id in range(self.num_pois):
            if poi_id < 6:
                self.pois[poi_id, 2] = -2.0
            else:
                self.pois[poi_id, 2] = 10.0

    def init_poi_vals_random_inner_square_outer(self):
        for poi_id in range(4):
            self.pois[poi_id, 2] = 100.0

        for poi_id in range(4, self.num_pois):
            self.pois[poi_id, 2] = 5.0

    def init_poi_vals_four_corners(self):
        assert(self.num_pois == 4)
        for poi_id in range(self.num_pois):
            if poi_id == 0:
                self.pois[poi_id, 2] = 2.0
            elif poi_id == 1:
                self.pois[poi_id, 2] = 5.0
            elif poi_id == 2:
                self.pois[poi_id, 2] = 6.0
            else:
                self.pois[poi_id, 2] = 12.0

    def init_poi_vals_clusters(self):
        assert(self.num_pois == 5)

        # First Cluster
        self.pois[0, 2] = 3.0
        self.pois[1, 2] = 3.0
        self.pois[2, 2] = 3.0

        # Others
        self.pois[3, 2] = 8.0
        self.pois[4, 2] = 9.0
