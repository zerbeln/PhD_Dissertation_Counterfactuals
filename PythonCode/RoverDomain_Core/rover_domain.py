import numpy as np
import math
import random
import os
import csv
import copy
from parameters import parameters as p
from RoverDomain_Core.agent import Poi, Rover


class RoverDomain:
    def __init__(self):
        # World attributes
        self.world_x = p["x_dim"]
        self.world_y = p["y_dim"]
        self.num_pois = p["n_poi"]
        self.n_rovers = p["n_rovers"]
        self.delta_min = p["min_distance"]  # Lower bound for distance in sensor/utility calculations
        self.obs_radius = p["observation_radius"]  # Maximum distance rovers can make observations of POI at

        # Rover Information
        self.initial_rover_positions = np.zeros((self.n_rovers, 3))
        self.rovers = {}  # Dictionary containing instances of rover objects

        # POI Information
        self.pois_info = np.zeros((self.num_pois, 5))
        self.pois = {}  # Dictionary containing instances of PoI objects

    def create_world_setup(self):
        """
        Create a new rover configuration file
        """

        # Initialize POI positions and values
        self.pois_info = np.zeros((self.num_pois, 5))  # [X, Y, Val, Coupling, Quadrant]
        if p["poi_config_type"] == "Random":
            self.poi_pos_random()
            self.poi_vals_random(3, 10)
        elif p["poi_config_type"] == "Two_POI":
            self.poi_pos_two_poi()
            self.poi_vals_identical(10.0)
        elif p["poi_config_type"] == "Four_Corners":
            self.poi_pos_four_corners()
            self.poi_vals_random(3.0, 10.0)
        elif p["poi_config_type"] == "Columns":
            self.poi_pos_columns()
            self.poi_vals_random(3.0, 10.0)
        elif p["poi_config_type"] == "Circle":
            self.poi_pos_circle()
            self.poi_vals_random(3.0, 10.0)
        elif p["poi_config_type"] == "Con_Circle":
            self.poi_pos_concentric_circles()
            self.poi_vals_random(3.0, 10.0)
        elif p["poi_config_type"] == "Four_Quadrants":
            self.poi_pos_three_quadrants()
            self.poi_vals_three_quadrant()
        else:
            print("ERROR, WRONG POI CONFIG KEY")
        self.save_poi_configuration()

        # Initialize Rover Positions
        self.initial_rover_positions = np.zeros((self.n_rovers, 3))  # [X, Y, Theta]
        if p["rover_config_type"] == "Random":
            self.rover_pos_random()
        elif p["rover_config_type"] == "Concentrated":
            self.rover_pos_center_concentrated()
        elif p["rover_config_type"] == "Four_Quadrants":
            self.rover_pos_quadrant_zero()
        self.save_rover_configuration()

    def load_world(self):
        """
        Load a rover domain from a saved configuration file
        """
        # Initialize POI positions and values
        self.load_poi_configuration()

        # Initialize Rover Positions
        self.load_rover_configuration()

    def calc_global(self):
        """
        Calculate the global reward at the current time step
        """
        global_reward = np.zeros(self.num_pois)

        for pk in self.pois:
            observer_count = 0
            rover_distances = copy.deepcopy(self.pois[pk].observer_distances)
            rover_distances = np.sort(rover_distances)  # Arranges distances from least to greatest

            for c in range(self.n_rovers):
                dist = rover_distances[c]
                if dist < self.obs_radius:
                    observer_count += 1

            # Update global reward if POI is observed
            if observer_count >= int(self.pois[pk].coupling):
                summed_dist = sum(rover_distances[0:int(self.pois[pk].coupling)])
                if self.pois[pk].hazardous:
                    global_reward[self.pois[pk].poi_id] = -10.0
                else:
                    global_reward[self.pois[pk].poi_id] = self.pois[pk].value / (summed_dist/self.pois[pk].coupling)

        return global_reward

    def save_poi_configuration(self):
        """
        Saves world configuration to a csv file in a folder called World_Config
        """
        dir_name = 'World_Config'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        pfile_name = os.path.join(dir_name, 'POI_Config.csv')

        with open(pfile_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for poi_id in range(self.num_pois):
                writer.writerow(self.pois_info[poi_id, :])

        csvfile.close()

    def load_poi_configuration(self):
        """
        Load in a save POI configuation from a CSV file
        """
        config_input = []
        with open('World_Config/POI_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        for poi_id in range(self.num_pois):
            poi_x = float(config_input[poi_id][0])
            poi_y = float(config_input[poi_id][1])
            poi_val = float(config_input[poi_id][2])
            poi_coupling = float(config_input[poi_id][3])
            poi_quadrant = float(config_input[poi_id][4])

            self.pois["P{0}".format(poi_id)] = Poi(poi_x, poi_y, poi_val, poi_coupling, poi_quadrant, poi_id)

    def save_rover_configuration(self):
        """
        Saves rover positions to a csv file in a folder called World_Config
        """
        dir_name = 'World_Config'  # Intended directory for output files

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

        csvfile.close()

    def load_rover_configuration(self):
        """
        Load initial rover position from saved csvfile
        """
        config_input = []
        with open('World_Config/Rover_Config.csv') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')

            for row in csv_reader:
                config_input.append(row)

        for rover_id in range(self.n_rovers):
            rov_x = float(config_input[rover_id][0])
            rov_y = float(config_input[rover_id][1])
            rov_theta = float(config_input[rover_id][2])

            self.rovers["R{0}".format(rover_id)] = Rover(rover_id, rov_x, rov_y, rov_theta)

    # ROVER POSITION FUNCTIONS ---------------------------------------------------------------------------------------
    def rover_pos_random(self):  # Randomly set rovers on map
        """
        Rovers given random starting positions and orientations
        """

        for rov_id in range(self.n_rovers):
            rover_x = random.uniform(0.0, self.world_x-1.0)
            rover_y = random.uniform(0.0, self.world_y-1.0)
            rover_theta = random.uniform(0.0, 360.0)

            rover_too_close = True
            while rover_too_close:
                count = 0
                for poi_id in range(self.num_pois):
                    x_dist = self.pois_info[poi_id, 0] - rover_x
                    y_dist = self.pois_info[poi_id, 1] - rover_y
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

    def rover_pos_center_concentrated(self):
        """
        Rovers given random starting positions within a radius of the center. Starting orientations are random
        """
        radius = 8.0
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

    def rover_pos_fixed_middle(self):  # Set rovers to fixed starting position
        """
        Create a starting position for the rover near the center of the world
        """
        for rov_id in range(self.n_rovers):
            self.initial_rover_positions[rov_id, 0] = 0.5*self.world_x
            self.initial_rover_positions[rov_id, 1] = 0.5*self.world_y
            self.initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)

    def rover_pos_quadrant_zero(self):
        """
        Initialize rover team in the center of Quadrant 0
        """
        radius = 3.0
        center_x = 3 * self.world_x / 4.0
        center_y = 3 * self.world_y / 4.0

        for rov_id in range(self.n_rovers):
            x = random.uniform(0.0, self.world_x - 1.0)  # Rover X-Coordinate
            y = random.uniform(0.0, self.world_y - 1.0)  # Rover Y-Coordinate

            while x > (center_x + radius) or x < (center_x - radius):
                x = random.uniform(0.0, self.world_x - 1.0)  # Rover X-Coordinate
            while y > (center_y + radius) or y < (center_y - radius):
                y = random.uniform(0.0, self.world_y - 1.0)  # Rover Y-Coordinate

            self.initial_rover_positions[rov_id, 0] = x  # Rover X-Coordinate
            self.initial_rover_positions[rov_id, 1] = y  # Rover Y-Coordinate
            self.initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)  # Rover orientation

    def rover_pos_quadrant_one(self):
        """
        Initialize rover team in the center of Quadrant 1
        """
        radius = 3.0
        center_x = self.world_x / 4.0
        center_y = 3 * self.world_y / 4.0

        for rov_id in range(self.n_rovers):
            x = random.uniform(0.0, self.world_x - 1.0)  # Rover X-Coordinate
            y = random.uniform(0.0, self.world_y - 1.0)  # Rover Y-Coordinate

            while x > (center_x + radius) or x < (center_x - radius):
                x = random.uniform(0.0, self.world_x - 1.0)  # Rover X-Coordinate
            while y > (center_y + radius) or y < (center_y - radius):
                y = random.uniform(0.0, self.world_y - 1.0)  # Rover Y-Coordinate

            self.initial_rover_positions[rov_id, 0] = x  # Rover X-Coordinate
            self.initial_rover_positions[rov_id, 1] = y  # Rover Y-Coordinate
            self.initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)  # Rover orientation

    def rover_pos_quadrant_two(self):
        """
        Initialize rover team in the center of Quadrant 2
        """
        radius = 3.0
        center_x = self.world_x / 4.0
        center_y = self.world_y / 4.0

        for rov_id in range(self.n_rovers):
            x = random.uniform(0.0, self.world_x - 1.0)  # Rover X-Coordinate
            y = random.uniform(0.0, self.world_y - 1.0)  # Rover Y-Coordinate

            while x > (center_x + radius) or x < (center_x - radius):
                x = random.uniform(0.0, self.world_x - 1.0)  # Rover X-Coordinate
            while y > (center_y + radius) or y < (center_y - radius):
                y = random.uniform(0.0, self.world_y - 1.0)  # Rover Y-Coordinate

            self.initial_rover_positions[rov_id, 0] = x  # Rover X-Coordinate
            self.initial_rover_positions[rov_id, 1] = y  # Rover Y-Coordinate
            self.initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)  # Rover orientation

    def rover_pos_quadrant_three(self):
        """
        Initialize rover team in the center of Quadrant 3
        """
        radius = 3.0
        center_x = 3 * self.world_x / 4.0
        center_y = self.world_y / 4.0

        for rov_id in range(self.n_rovers):
            x = random.uniform(0.0, self.world_x - 1.0)  # Rover X-Coordinate
            y = random.uniform(0.0, self.world_y - 1.0)  # Rover Y-Coordinate

            while x > (center_x + radius) or x < (center_x - radius):
                x = random.uniform(0.0, self.world_x - 1.0)  # Rover X-Coordinate
            while y > (center_y + radius) or y < (center_y - radius):
                y = random.uniform(0.0, self.world_y - 1.0)  # Rover Y-Coordinate

            self.initial_rover_positions[rov_id, 0] = x  # Rover X-Coordinate
            self.initial_rover_positions[rov_id, 1] = y  # Rover Y-Coordinate
            self.initial_rover_positions[rov_id, 2] = random.uniform(0.0, 360.0)  # Rover orientation

    # POI POSITION FUNCTIONS ------------------------------------------------------------------------------------------
    def poi_pos_random(self):  # Randomly set POI on the map
        """
        POI positions set randomly across the map (but not in range of any rover)
        """
        for poi_id in range(self.num_pois):
            x = random.uniform(0, self.world_x-1.0)
            y = random.uniform(0, self.world_y-1.0)

            poi_too_close = True
            while poi_too_close:
                count = 0
                for p_id in range(self.num_pois):
                    if p_id != poi_id:
                        x_dist = x - self.pois_info[p_id, 0]
                        y_dist = y - self.pois_info[p_id, 1]

                        dist = math.sqrt((x_dist**2) + (y_dist**2))
                        if dist < (self.obs_radius + 2):
                            count += 1

                if count == 0:
                    poi_too_close = False
                else:
                    x = random.uniform(0, self.world_x - 1.0)
                    y = random.uniform(0, self.world_y - 1.0)
                    count = 0

            self.pois_info[poi_id, 0] = x
            self.pois_info[poi_id, 1] = y
            self.pois_info[poi_id, 3] = 1

            angle = self.get_angle(self.pois_info[poi_id, 0], self.pois_info[poi_id, 1])
            q = int(angle/p["angle_res"])
            self.pois_info[poi_id, 4] = q

    def poi_pos_circle(self):
        """
        POI positions are set in a circle around the center of the map at a specified radius.
        """
        radius = 15.0
        interval = float(360/self.num_pois)

        x = self.world_x/2.0
        y = self.world_y/2.0
        theta = 0.0

        for poi_id in range(self.num_pois):
            self.pois_info[poi_id, 0] = x + radius*math.cos(theta*math.pi/180)
            self.pois_info[poi_id, 1] = y + radius*math.sin(theta*math.pi/180)
            self.pois_info[poi_id, 3] = 1
            angle = self.get_angle(self.pois_info[poi_id, 0], self.pois_info[poi_id, 1])
            q = int(angle / p["angle_res"])
            self.pois_info[poi_id, 4] = q
            theta += interval

    def poi_pos_concentric_circles(self):
        """
        POI positions are set in a circle around the center of the map at a specified radius.
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
                self.pois_info[poi_id, 0] = x + inner_radius * math.cos(inner_theta * math.pi / 180)
                self.pois_info[poi_id, 1] = y + inner_radius * math.sin(inner_theta * math.pi / 180)
                self.pois_info[poi_id, 3] = 1
                angle = self.get_angle(self.pois_info[poi_id, 0], self.pois_info[poi_id, 1])
                q = int(angle / p["angle_res"])
                self.pois_info[poi_id, 4] = q
                inner_theta += interval
            else:
                self.pois_info[poi_id, 0] = x + outter_radius * math.cos(outter_theta * math.pi / 180)
                self.pois_info[poi_id, 1] = y + outter_radius * math.sin(outter_theta * math.pi / 180)
                self.pois_info[poi_id, 3] = 1
                angle = self.get_angle(self.pois_info[poi_id, 0], self.pois_info[poi_id, 1])
                q = int(angle / p["angle_res"])
                self.pois_info[poi_id, 4] = q
                outter_theta += interval

    def poi_pos_three_quadrants(self):
        """
        For world with 5 POI, three in Quadrant 1, 1 in Quadrant 2, 1 in Quadrant 3
        """
        assert(self.num_pois == 5)

        self.pois_info[0, 0] = 10
        self.pois[0, 1] = 10

        self.pois_info[1, 0] = 12
        self.pois_info[1, 1] = 12

        self.pois_info[2, 0] = 8
        self.pois_info[2, 1] = 11

        self.pois_info[3, 0] = self.world_x - 5
        self.pois_info[3, 1] = self.world_y - 5

        self.pois_info[4, 0] = (self.world_x/2) - 8
        self.pois_info[4, 1] = (self.world_y/2) + 10

        for poi_id in range(self.num_pois):
            self.pois_info[poi_id, 3] = 1
            angle = self.get_angle(self.pois_info[poi_id, 0], self.pois_info[poi_id, 1])
            q = int(angle / p["angle_res"])
            self.pois_info[poi_id, 4] = q

    def poi_pos_two_poi(self):
        """
        Sets two POI on the map, one on the left, one on the right at Y-Dimension/2
        """
        assert(self.num_pois == 2)

        # Left POI
        self.pois_info[0, 0] = 1.0
        self.pois_info[0, 1] = (self.world_y/2.0) - 1
        self.pois_info[0, 3] = 1
        angle = self.get_angle(self.pois_info[0, 0], self.pois_info[0, 1])
        self.pois_info[0, 4] = int(angle / p["angle_res"])

        # Right POI
        self.pois_info[1, 0] = self.world_x - 2.0
        self.pois_info[1, 1] = (self.world_y/2.0) + 1
        self.pois_info[1, 3] = 1
        angle = self.get_angle(self.pois_info[1, 0], self.pois_info[1, 1])
        self.pois_info[1, 4] = int(angle / p["angle_res"])

    def poi_pos_four_corners(self):  # Statically set 4 POI (one in each corner)
        """
        Sets 4 POI on the map in a box formation around the center
        """
        assert(self.num_pois == 4)  # There must only be 4 POI for this initialization

        # Bottom left
        self.pois_info[0, 0] = 2.0
        self.pois_info[0, 1] = 2.0
        self.pois_info[0, 3] = 1
        self.pois_info[0, 4] = 1

        # Top left
        self.pois_info[1, 0] = 2.0
        self.pois_info[1, 1] = (self.world_y - 2.0)
        self.pois_info[1, 3] = 1
        self.pois_info[1, 4] = 2

        # Bottom right
        self.pois_info[2, 0] = (self.world_x - 2.0)
        self.pois_info[2, 1] = 2.0
        self.pois_info[2, 3] = 1
        self.pois_info[2, 4] = 0

        # Top right
        self.pois_info[3, 0] = (self.world_x - 2.0)
        self.pois_info[3, 1] = (self.world_y - 2.0)
        self.pois_info[3, 3] = 1
        self.pois_info[3, 4] = 3

    def poi_pos_columns(self):
        """
        Sets 6 POI in two columns on opposite sides of the map
        """

        assert (self.num_pois == 6)  # There must only be 4 POI for this initialization

        # Bottom left
        self.pois_info[0, 0] = 2.0
        self.pois_info[0, 1] = 2.0
        self.pois_info[0, 3] = 1
        self.pois_info[0, 4] = 1

        # Middle Left
        self.pois_info[1, 0] = 2.0
        self.pois_info[1, 1] = self.world_y/2
        self.pois_info[1, 3] = 1
        self.pois_info[1, 4] = 1

        # Top left
        self.pois_info[2, 0] = 2.0
        self.pois_info[2, 1] = (self.world_y - 2.0)
        self.pois_info[2, 3] = 1
        self.pois_info[2, 4] = 2

        # Bottom right
        self.pois_info[3, 0] = (self.world_x - 2.0)
        self.pois_info[3, 1] = 2.0
        self.pois_info[3, 3] = 1
        self.pois_info[3, 4] = 0

        # Middle Right
        self.pois_info[4, 0] = (self.world_x - 2.0)
        self.pois_info[4, 1] = self.world_y/2
        self.pois_info[4, 3] = 1
        self.pois_info[4, 4] = 0

        # Top right
        self.pois_info[5, 0] = (self.world_x - 2.0)
        self.pois_info[5, 1] = (self.world_y - 2.0)
        self.pois_info[5, 3] = 1
        self.pois_info[5, 4] = 3

    # POI VALUE FUNCTIONS -----------------------------------------------------------------------------------
    def poi_vals_random(self, v_low, v_high):
        """
        POI values randomly assigned 1-10
        """

        for poi_id in range(self.num_pois):
            self.pois_info[poi_id, 2] = float(random.randint(v_low, v_high))

    def poi_vals_identical(self, poi_val):
        """
        POI values set to fixed, identical value
        """

        for poi_id in range(self.num_pois):
            self.pois_info[poi_id, 2] = poi_val

    def poi_vals_three_quadrant(self):

        # Quadrant 2 (Three Low Value)
        self.pois_info[0, 2] = 10.0
        self.pois_info[1, 2] = 10.0
        self.pois_info[2, 2] = 10.0

        # Quadrant 1 (Single High Value)
        self.pois_info[3, 2] = 100.0

        # Quadrant 2 (Single Low Value)
        self.pois_info[4, 2] = 5.0

    def get_angle(self, px, py):
        """
        Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        """
        origin_x = self.world_x / 2
        origin_y = self.world_y / 2
        x = px - origin_x
        y = py - origin_y

        angle = math.atan2(y, x)*(180.0/math.pi)

        while angle < 0.0:
            angle += 360.0
        while angle > 360.0:
            angle -= 360.0

        return angle

