import numpy as np
from parameters import parameters as p
import sys
from global_functions import get_angle, get_squared_dist


class Supervisor:
    def __init__(self):
        # Rover Sensor Characteristics -----------------------------------------------------------------------
        self.sensor_type = p["sensor_model"]  # Type of sensors rover is equipped with
        self.sensor_range = None  # Distance rovers can perceive environment (default is infinite)
        self.sensor_res = p["angle_res"]  # Angular resolution of the sensors

        # Supervisor Data -----------------------------------------------------------------------------------------
        self.observations = np.zeros(p["n_inp"], dtype=np.longdouble)  # Number of sensor inputs for Neural Network

    def scan_environment(self, rovers, pois):
        """
        Constructs the state information that gets passed to the rover's neuro-controller
        :param rovers: Dictionary containing rover class instances
        :param pois: Dictionary containing POI class instances
        """
        n_brackets = int(360.0 / self.sensor_res)
        poi_state = self.poi_scan(pois, n_brackets)
        rover_state = self.rover_scan(rovers, n_brackets)

        for i in range(n_brackets):
            self.observations[i] = poi_state[i]
            self.observations[n_brackets + i] = rover_state[i]

    def poi_scan(self, pois, n_brackets):
        """
        Supervisor observes POIs in the environment using sensors
        :param pois: Dictionary containing POI class instances
        :param n_brackets: integer value for the number of brackets/sectors rover sensors scan (resolution)
        :return poi_state: numpy array containing state information for POI observations
        """
        poi_state = np.zeros(n_brackets)
        temp_poi_dist_list = [[] for _ in range(n_brackets)]

        # Log POI distances into brackets
        poi_id = 0
        for poi in pois:
            angle = get_angle(pois[poi].loc[0], pois[poi].loc[1], (p["x_dim"]/2), (p["y_dim"]/2))
            dist = get_squared_dist(pois[poi].loc[0], pois[poi].loc[1], (p["x_dim"]/2), (p["y_dim"]/2))

            bracket = int(angle / self.sensor_res)
            if bracket > n_brackets-1:
                bracket -= n_brackets
            temp_poi_dist_list[bracket].append(pois[poi].value / (pois[poi].coupling*dist))
            poi_id += 1

        # Encode POI information into the state vector
        for bracket in range(n_brackets):
            num_poi_bracket = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
            if num_poi_bracket > 0:
                if self.sensor_type == 'density':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi_bracket  # Density Sensor
                elif self.sensor_type == 'summed':
                    poi_state[bracket] = sum(temp_poi_dist_list[bracket])  # Summed Distance Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                poi_state[bracket] = -1.0

        return poi_state

    def rover_scan(self, rovers, n_brackets):
        """
        Supervisor observes other rovers in the environment using sensors
        :param rovers: Dictionary containing rover class instances
        :param n_brackets: integer value for the number of brackets/sectors rover sensors scan (resolution)
        :return rover_state: numpy array containing state information for Rover observations
        """
        rover_state = np.zeros(n_brackets)
        temp_rover_dist_list = [[] for _ in range(n_brackets)]

        # Log Rover distances into brackets
        for rv in rovers:
            rov_x = rovers[rv].loc[0]
            rov_y = rovers[rv].loc[1]

            angle = get_angle(rov_x, rov_y, p["x_dim"]/2, p["y_dim"]/2)
            dist = get_squared_dist(rov_x, rov_y, (p["x_dim"]/2), (p["y_dim"]/2))

            bracket = int(angle / self.sensor_res)
            if bracket > n_brackets-1:
                bracket -= n_brackets
            temp_rover_dist_list[bracket].append(1 / dist)

        # Encode Rover information into the state vector
        for bracket in range(n_brackets):
            num_rovers_bracket = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
            if num_rovers_bracket > 0:
                if self.sensor_type == 'density':
                    rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_rovers_bracket  # Density Sensor
                elif self.sensor_type == 'summed':
                    rover_state[bracket] = sum(temp_rover_dist_list[bracket])  # Summed Distance Sensor
                else:
                    sys.exit('Incorrect sensor model')
            else:
                rover_state[bracket] = -1.0

        return rover_state
