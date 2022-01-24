from suggestion_network import SuggestionNetwork
from RoverDomain_Core.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
import pickle
import csv
import os
import sys
import math
import numpy as np
from parameters import parameters as p


def save_reward_history(reward_history, file_name):
    """
    Save reward data as a CSV file for graph generation. CSV is appended each time function is called.
    """

    dir_name = 'Output_Data/'  # Intended directory for output files
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, file_name)
    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(reward_history)


def save_rover_path(rover_path, file_name):  # Save path rovers take using best policy found
    """
    Records the path each rover takes using best policy from CCEA (used by visualizer)
    """
    dir_name = 'Output_Data/'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    rpath_name = os.path.join(dir_name, file_name)
    rover_file = open(rpath_name, 'wb')
    pickle.dump(rover_path, rover_file)
    rover_file.close()


def load_saved_policies(file_name, rover_id, srun):
    """
    Load saved Neural Network policies from pickle file
    """

    dir_name = 'Policy_Bank/Rover{0}/SRUN{1}'.format(rover_id, srun)
    fpath_name = os.path.join(dir_name, file_name)
    weight_file = open(fpath_name, 'rb')
    weights = pickle.load(weight_file)
    weight_file.close()

    return weights


def create_policy_bank(playbook_type, rover_id, srun):
    """
    Choose which playbook of policies to load for the rovers
    """
    policy_bank = {}

    if playbook_type == "Four_Quadrants":
        w0 = load_saved_policies("TowardQuadrant0", rover_id, srun)
        w1 = load_saved_policies("TowardQuadrant1", rover_id, srun)
        w2 = load_saved_policies("TowardQuadrant2", rover_id, srun)
        w3 = load_saved_policies("TowardQuadrant3", rover_id, srun)
        policy_bank["Policy0"] = w0
        policy_bank["Policy1"] = w1
        policy_bank["Policy2"] = w2
        policy_bank["Policy3"] = w3
    elif playbook_type == "Two_POI":
        w0 = load_saved_policies("TowardPOI0", rover_id, srun)
        w1 = load_saved_policies("TowardPOI1", rover_id, srun)
        policy_bank["Policy0"] = w0
        policy_bank["Policy1"] = w1

    return policy_bank


def get_angle_dist(x, y, tx, ty):
    """
    Computes angles and distance between two predators relative to (1,0) vector (x-axis)
    :param tx: X-Position of sensor target
    :param ty: Y-Position of sensor target
    :param x: X-Position of scanning rover
    :param y: Y-Position of scanning rover
    :return: angle, dist
    """

    vx = x - tx
    vy = y - ty

    angle = math.atan2(vy, vx)*(180.0/math.pi)

    while angle < 0.0:
        angle += 360.0
    while angle > 360.0:
        angle -= 360.0
    if math.isnan(angle):
        angle = 0.0

    dist = (vx**2) + (vy**2)

    # Clip distance to not overwhelm activation function in NN
    if dist < p["dmax"]:
        dist = p["dmax"]

    return angle, dist


def construct_counterfactual_state(pois, rovers, rover_id, suggestion):
    """
    Create a counteractual state input to represent agent suggestions
    """

    rx = rovers["R{0}".format(rover_id)].x_pos
    ry = rovers["R{0}".format(rover_id)].y_pos
    cfact_poi = create_counterfactual_poi_state(pois, rx, ry, suggestion)
    cfact_rover = create_counterfactual_rover_state(rovers, rx, ry, rover_id, suggestion)

    counterfactual_state = np.zeros(8)
    for i in range(4):
        counterfactual_state[i] = cfact_poi[i]
        counterfactual_state[4 + i] = cfact_rover[i]

    return counterfactual_state


def create_counterfactual_poi_state(pois, rx, ry, suggestion):
    """
    Construct a counterfactual state input for POI detections
    :return: Portion of the counterfactual state constructed from POI scanner
    """
    c_poi_state = np.zeros(int(360.0 / p["angle_res"]))
    temp_poi_dist_list = [[] for _ in range(int(360.0 / p["angle_res"]))]

    # Log POI distances into brackets
    for poi in pois:
        angle, dist = get_angle_dist(rx, ry, pois[poi].x_position, pois[poi].y_position)

        bracket = int(angle / p["angle_res"])
        if bracket > 3:
            bracket -= 4
        if pois[poi].quadrant == suggestion:
            temp_poi_dist_list[bracket].append(10*pois[poi].value / dist)

    # Encode POI information into the state vector
    for bracket in range(int(360 / p["angle_res"])):
        num_poi_bracket = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
        if num_poi_bracket > 0:
            if p["sensor_model"] == 'density':
                c_poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi_bracket  # Density Sensor
            elif p["sensor_model"] == 'summed':
                c_poi_state[bracket] = sum(temp_poi_dist_list[bracket])  # Summed Distance Sensor
            else:
                sys.exit('Incorrect sensor model')
        else:
            c_poi_state[bracket] = -1.0

    return c_poi_state


def create_counterfactual_rover_state(rovers, rx, ry, rover_id, suggestion):
    """
    Construct a counterfactual state input for rover detections
    :return: Portion of the counterfactual state vector created from rover scanner
    """
    center_x = p["x_dim"]/2
    center_y = p["y_dim"]/2
    rover_state = np.zeros(int(360.0 / p["angle_res"]))
    temp_rover_dist_list = [[] for _ in range(int(360.0 / p["angle_res"]))]

    # Log rover distances into brackets
    for r in rovers:
        if rovers[r].self_id != rover_id:  # Ignore self
            rov_x = rovers[r].x_pos
            rov_y = rovers[r].y_pos

            angle, dist = get_angle_dist(rx, ry, rov_x, rov_y)
            bracket = int(angle / p["angle_res"])
            if bracket > 3:
                bracket -= 4

            w_angle, w_dist = get_angle_dist(center_x, center_y, rov_x, rov_y)
            world_bracket = int(w_angle/p["angle_res"])
            if world_bracket > 3:
                world_bracket -= 4

            if suggestion == world_bracket:
                temp_rover_dist_list[bracket].append(10 / dist)

    # Encode Rover information into the state vector
    for bracket in range(int(360 / p["angle_res"])):
        num_rovers_bracket = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
        if num_rovers_bracket > 0:
            if p["sensor_model"] == 'density':
                rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_rovers_bracket  # Density Sensor
            elif p["sensor_model"] == 'summed':
                rover_state[bracket] = sum(temp_rover_dist_list[bracket])  # Summed Distance Sensor
            else:
                sys.exit('Incorrect sensor model')
        else:
            rover_state[bracket] = -1.0

    return rover_state


def test_suggestions_policy_bank(pbank_type, sgst):
    """
    Test suggestions using the pre-trained policy bank
    """
    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # Suggestion Parameters
    s_inp = p["s_inputs"]
    s_hid = p["s_hidden"]
    s_out = p["s_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["SN{0}".format(rover_id)] = SuggestionNetwork(s_inp, s_out, s_hid)

    average_reward = 0
    reward_history = []  # Keep track of team performance throughout training
    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs

        # Load Trained Suggestion Interpreter Weights
        for rover_id in range(n_rovers):
            rd.rovers["R{0}".format(rover_id)].policy_bank = create_policy_bank(pbank_type, rover_id, srun)
            s_weights = load_saved_policies('SelectionWeights{0}'.format(rover_id), rover_id, srun)
            pops["SN{0}".format(rover_id)].get_weights(s_weights)

        for rov in rd.rovers:
            rd.rovers[rov].reset_rover()
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 0] = rd.rovers[rov].x_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 1] = rd.rovers[rov].y_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 2] = rd.rovers[rov].theta_pos

        for rover_id in range(n_rovers):  # Initial rover scan of environment
            suggestion = construct_counterfactual_state(rd.pois, rd.rovers, rover_id, sgst[rover_id])
            rd.rovers["R{0}".format(rover_id)].scan_environment(rd.rovers, rd.pois)
            sensor_data = rd.rovers["R{0}".format(rover_id)].sensor_readings
            sug_input = np.concatenate((suggestion, sensor_data), axis=0)
            pops["SN{0}".format(rover_id)].get_inputs(sug_input)

            # Determine action based on sensor inputs and suggestion
            sug_outputs = pops["SN{0}".format(rover_id)].get_outputs()
            pol_id = np.argmax(sug_outputs)
            weights = rd.rovers["R{0}".format(rover_id)].policy_bank["Policy{0}".format(pol_id)]
            rd.rovers["R{0}".format(rover_id)].get_weights(weights)
            rd.rovers["R{0}".format(rover_id)].get_nn_outputs()

        g_rewards = np.zeros(rover_steps)
        for step_id in range(rover_steps):
            # Rover takes an action in the world
            for rov in rd.rovers:
                rd.rovers[rov].step(rd.world_x, rd.world_y)
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 0] = rd.rovers[rov].x_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 1] = rd.rovers[rov].y_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 2] = rd.rovers[rov].theta_pos

            # Rover scans environment and processes suggestions
            for rover_id in range(n_rovers):
                suggestion = construct_counterfactual_state(rd.pois, rd.rovers, rover_id, sgst[rover_id])
                rd.rovers["R{0}".format(rover_id)].scan_environment(rd.rovers, rd.pois)
                rd.update_observer_distances()
                sensor_data = rd.rovers["R{0}".format(rover_id)].sensor_readings
                pops["SN{0}".format(rover_id)].get_inputs(np.concatenate((suggestion, sensor_data), axis=0))

                # Determine action based on sensor inputs and suggestion
                sug_outputs = pops["SN{0}".format(rover_id)].get_outputs()
                pol_id = np.argmax(sug_outputs)
                weights = rd.rovers["R{0}".format(rover_id)].policy_bank["Policy{0}".format(pol_id)]
                rd.rovers["R{0}".format(rover_id)].get_weights(weights)
                rd.rovers["R{0}".format(rover_id)].get_nn_outputs()

            # Calculate Global Reward
            g_rewards[step_id] = rd.calc_global()

        reward_history.append(sum(g_rewards))
        average_reward += sum(g_rewards)

        save_rover_path(final_rover_path, "Rover_Paths")

    average_reward /= stat_runs
    print(average_reward)
    save_reward_history(reward_history, "Final_GlobalRewards.csv")
    run_visualizer()


if __name__ == '__main__':

    # Create list of suggestions for rovers to use during training
    rover_suggestions = []
    if p["policy_bank_type"] == "Two_POI":
        rover_suggestions = [1, 1, 1]
    elif p["policy_bank_type"] == "Four_Quadrants":
        rover_suggestions = [1, 2, 0, 0, 3, 0]
    test_suggestions_policy_bank(p["policy_bank_type"], rover_suggestions)

