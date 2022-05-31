from parameters import parameters as p
from global_functions import load_saved_policies, get_angle, get_squared_dist
import numpy as np
import sys


def create_policy_bank(playbook_type, rover_id, srun):
    """
    Load the pre-trained rover skills into a dictionary from saved pickle files
    """
    policy_bank = {}

    if playbook_type == "Target_Quadrant":
        for q_id in range(4):
            w = load_saved_policies("TowardQuadrant{0}".format(q_id), rover_id, srun)
            policy_bank["Policy{0}".format(q_id)] = w
    elif playbook_type == "Target_POI":
        for poi_id in range(p["n_poi"]):
            w = load_saved_policies("TowardPOI{0}".format(poi_id), rover_id, srun)
            policy_bank["Policy{0}".format(poi_id)] = w

    return policy_bank


def get_counterfactual_state(pois, rovers, rover_id, suggestion):
    """
    Create a counteractual state input to represent agent suggestions
    """
    n_brackets = int(360.0 / p["angle_res"])
    rx = rovers["R{0}".format(rover_id)].x_pos
    ry = rovers["R{0}".format(rover_id)].y_pos
    cfact_poi = create_counterfactual_poi_state(pois, rx, ry, n_brackets, suggestion)
    cfact_rover = create_counterfactual_rover_state(pois, rovers, rx, ry, n_brackets, rover_id, suggestion)

    counterfactual_state = np.zeros(int(n_brackets*2))
    for i in range(n_brackets):
        counterfactual_state[i] = cfact_poi[i]
        counterfactual_state[n_brackets + i] = cfact_rover[i]

    return counterfactual_state


def create_counterfactual_poi_state(pois, rx, ry, n_brackets, suggestion):
    """
    Construct a counterfactual state based on POI sensors
    """
    c_poi_state = np.zeros(n_brackets)
    temp_poi_dist_list = [[] for _ in range(n_brackets)]

    # Log POI distances into brackets
    for poi in pois:
        # angle = get_angle(pois[poi].x_position, pois[poi].y_position, rx, ry)
        angle = get_angle(pois[poi].x_position, pois[poi].y_position, (p["x_dim"]/2), (p["y_dim"]/2))
        dist = get_squared_dist(pois[poi].x_position, pois[poi].y_position, rx, ry)
        if dist < p["min_distance"]:
            dist = p["min_distance"]

        bracket = int(angle / p["angle_res"])
        if bracket > n_brackets-1:
            bracket -= n_brackets
        if pois[poi].poi_id == suggestion:  # This can also be switched from POI ID to POI Quadrant
            temp_poi_dist_list[bracket].append(pois[poi].value/dist)
        else:
            temp_poi_dist_list[bracket].append(-1 * pois[poi].value/dist)

    # Encode POI information into the state vector
    for bracket in range(n_brackets):
        num_poi_bracket = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
        if num_poi_bracket > 0:
            if p["sensor_model"] == 'density':
                c_poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi_bracket  # Density Sensor
            elif p["sensor_model"] == 'summed':
                c_poi_state[bracket] = sum(temp_poi_dist_list[bracket])  # Summed Distance Sensor
            else:
                sys.exit('Incorrect sensor model')
        else:
            c_poi_state[bracket] = 0.0

    return c_poi_state


def create_counterfactual_rover_state(pois, rovers, rx, ry, n_brackets, rover_id, suggestion):
    """
    Construct a counterfactual state input based on Rover sensors
    """
    rover_state = np.zeros(n_brackets)
    temp_rover_dist_list = [[] for _ in range(n_brackets)]

    poi_quadrant = pois["P{0}".format(suggestion)].quadrant

    # Log rover distances into brackets
    for r in rovers:
        if rovers[r].self_id != rover_id:  # Ignore self
            rov_x = rovers[r].x_pos
            rov_y = rovers[r].y_pos

            # angle = get_angle(rov_x, rov_y, rx, ry)
            angle = get_angle(rov_x, rov_y, (p["x_dim"]/2), (p["y_dim"]/2))
            dist = get_squared_dist(rov_x, rov_y, rx, ry)
            if dist < p["min_distance"]:
                dist = p["min_distance"]

            bracket = int(angle / p["angle_res"])
            if bracket > n_brackets-1:
                bracket -= n_brackets

            if poi_quadrant == bracket:
                temp_rover_dist_list[bracket].append(0/dist)
            else:
                temp_rover_dist_list[bracket].append(0/dist)

    # Encode Rover information into the state vector
    for bracket in range(n_brackets):
        num_rovers_bracket = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
        if num_rovers_bracket > 0:
            if p["sensor_model"] == 'density':
                rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_rovers_bracket  # Density Sensor
            elif p["sensor_model"] == 'summed':
                rover_state[bracket] = sum(temp_rover_dist_list[bracket])  # Summed Distance Sensor
            else:
                sys.exit('Incorrect sensor model')
        else:
            rover_state[bracket] = 0.0

    return rover_state
