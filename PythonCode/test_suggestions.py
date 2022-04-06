from suggestion_network import CBANetwork
from RoverDomain_Core.rover_domain import RoverDomain
from RewardFunctions.local_rewards import *
from Visualizer.visualizer import run_visualizer
import pickle
import csv
import os
import sys
import math
import numpy as np
import random
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

    if playbook_type == "Target_Quadrant":
        for q_id in range(4):
            w = load_saved_policies("TowardQuadrant{0}".format(q_id), rover_id, srun)
            policy_bank["Policy{0}".format(q_id)] = w
    elif playbook_type == "Target_POI":
        for poi_id in range(p["n_poi"]):
            w = load_saved_policies("TowardPOI{0}".format(poi_id), rover_id, srun)
            policy_bank["Policy{0}".format(poi_id)] = w

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
    Construct a counterfactual state input for POI detections
    """
    c_poi_state = np.zeros(n_brackets)
    temp_poi_dist_list = [[] for _ in range(n_brackets)]

    # Log POI distances into brackets
    for poi in pois:
        angle, dist = get_angle_dist(rx, ry, pois[poi].x_position, pois[poi].y_position)

        bracket = int(angle / p["angle_res"])
        if bracket > n_brackets-1:
            bracket -= n_brackets
        if pois[poi].poi_id == suggestion:  # This can also be switched from POI ID to POI Quadrant
            temp_poi_dist_list[bracket].append(pois[poi].value/dist)
        else:
            temp_poi_dist_list[bracket].append(-2 * pois[poi].value/dist)

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
            c_poi_state[bracket] = -1.0

    return c_poi_state


def create_counterfactual_rover_state(pois, rovers, rx, ry, n_brackets, rover_id, suggestion):
    """
    Construct a counterfactual state input for rover detections
    """
    center_x = p["x_dim"]/2
    center_y = p["y_dim"]/2
    rover_state = np.zeros(n_brackets)
    temp_rover_dist_list = [[] for _ in range(n_brackets)]

    poi_quadrant = pois["P{0}".format(suggestion)].quadrant

    # Log rover distances into brackets
    for r in rovers:
        if rovers[r].self_id != rover_id:  # Ignore self
            rov_x = rovers[r].x_pos
            rov_y = rovers[r].y_pos

            angle, dist = get_angle_dist(rx, ry, rov_x, rov_y)
            bracket = int(angle / p["angle_res"])
            if bracket > n_brackets-1:
                bracket -= n_brackets

            w_angle, w_dist = get_angle_dist(center_x, center_y, rov_x, rov_y)
            world_bracket = int(w_angle/p["angle_res"])
            if world_bracket > 3:
                world_bracket -= 4

            if poi_quadrant == world_bracket:
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
            rover_state[bracket] = -1.0

    return rover_state


def test_skill_performance(skill_id):
    """
    Test suggestions using the pre-trained policy bank
    """
    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    srun = p["starting_srun"]
    while srun < stat_runs:  # Perform statistical runs
        skill_performance = []  # Keep track of team performance throughout training
        # Load Trained Suggestion Interpreter Weights
        for rov in rd.rovers:
            rover_id = rd.rovers[rov].self_id
            if p["policy_bank_type"] == "Target_POI":
                weights = load_saved_policies("TowardPOI{0}".format(skill_id), rover_id, srun)
            elif p["policy_bank_type"] == "Target_Quadrant":
                weights = load_saved_policies("TowardQuadrant{0}".format(skill_id), rover_id, srun)
            rd.rovers[rov].get_weights(weights)
            rd.rovers[rov].reset_rover()
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 0] = rd.rovers[rov].x_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 1] = rd.rovers[rov].y_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 2] = rd.rovers[rov].theta_pos

        for rov in rd.rovers:  # Initial rover scan of environment
            rd.rovers[rov].scan_environment(rd.rovers, rd.pois)
        for poi in rd.pois:
            rd.pois[poi].update_observer_distances(rd.rovers)

        rewards = [[] for i in range(n_rovers)]
        for step_id in range(rover_steps):
            # Rover takes an action in the world
            for rov in rd.rovers:
                rd.rovers[rov].step(rd.world_x, rd.world_y)
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 0] = rd.rovers[rov].x_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 1] = rd.rovers[rov].y_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 2] = rd.rovers[rov].theta_pos

            # Rover scans environment and observer distances are updated
            for rk in rd.rovers:  # Rover scans environment
                rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
            for poi in rd.pois:
                rd.pois[poi].update_observer_distances(rd.rovers)

            # Calculate Global Reward
            for rover_id in range(n_rovers):
                if p["policy_bank_type"] == "Target_POI":
                    rewards[rover_id].append(target_poi_reward(rover_id, rd.pois, skill_id))
                elif p["policy_bank_type"] == "Target_Quadrant":
                    rewards[rover_id].append(target_quadrant_reward(rover_id, rd.pois, skill_id))


        for rover_id in range(n_rovers):
            skill_performance.append(sum(rewards[rover_id]))

        save_rover_path(final_rover_path, "Rover_Paths")
        save_reward_history(skill_performance, "Skill{0}_Performance.csv".format(skill_id))

        srun += 1


def test_suggestions_policy_bank(pbank_type, sgst):
    """
    Test suggestions using the pre-trained policy bank
    """
    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Generate Hazard Areas (If Testing For Hazards)
    if p["active_hazards"]:
        rd.pois["P0"].hazardous = True

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["CBA{0}".format(rover_id)] = CBANetwork()

    average_reward = 0
    reward_history = []  # Keep track of team performance throughout training
    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    srun = p["starting_srun"]
    while srun < stat_runs:  # Perform statistical runs
        # Load Trained Suggestion Interpreter Weights
        for rover_id in range(n_rovers):
            rd.rovers["R{0}".format(rover_id)].policy_bank = create_policy_bank(pbank_type, rover_id, srun)
            s_weights = load_saved_policies('SelectionWeights{0}'.format(rover_id), rover_id, srun)
            pops["CBA{0}".format(rover_id)].get_weights(s_weights)

        for rov in rd.rovers:
            rd.rovers[rov].reset_rover()
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 0] = rd.rovers[rov].x_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 1] = rd.rovers[rov].y_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 2] = rd.rovers[rov].theta_pos

        for rk in rd.rovers:  # Initial rover scan of environment
            rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
        for poi in rd.pois:
            rd.pois[poi].update_observer_distances(rd.rovers)
        for rk in rd.rovers:  # Initial rover scan of environment
            rover_id = rd.rovers[rk].self_id
            suggestion = construct_counterfactual_state(rd.pois, rd.rovers, rover_id, sgst[rover_id])
            sensor_data = rd.rovers[rk].sensor_readings
            cba_input = np.sum((suggestion, sensor_data), axis=0)
            pops["CBA{0}".format(rover_id)].get_inputs(cba_input)

            # Determine action based on sensor inputs and suggestion
            cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()
            pol_id = int(cba_outputs)
            weights = rd.rovers[rk].policy_bank["Policy{0}".format(pol_id)]
            rd.rovers[rk].get_weights(weights)
            rd.rovers[rk].get_nn_outputs()

        g_rewards = np.zeros(rover_steps)
        for step_id in range(rover_steps):
            # Rover takes an action in the world
            for rov in rd.rovers:
                rd.rovers[rov].step(rd.world_x, rd.world_y)
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 0] = rd.rovers[rov].x_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 1] = rd.rovers[rov].y_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 2] = rd.rovers[rov].theta_pos

            # Rover scans environment and processes suggestions
            for rk in rd.rovers:  # Rover scans environment
                rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
            for poi in rd.pois:
                rd.pois[poi].update_observer_distances(rd.rovers)
            for rover_id in range(n_rovers):
                suggestion = construct_counterfactual_state(rd.pois, rd.rovers, rover_id, sgst[rover_id])
                sensor_data = rd.rovers["R{0}".format(rover_id)].sensor_readings
                cba_input = np.sum((suggestion, sensor_data), axis=0)
                pops["CBA{0}".format(rover_id)].get_inputs(cba_input)

                # Determine action based on sensor inputs and suggestion
                cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()
                pol_id = int(cba_outputs)
                weights = rd.rovers["R{0}".format(rover_id)].policy_bank["Policy{0}".format(pol_id)]
                rd.rovers["R{0}".format(rover_id)].get_weights(weights)
                rd.rovers["R{0}".format(rover_id)].get_nn_outputs()

            # Calculate Global Reward
            g_rewards[step_id] = rd.calc_global()

        reward_history.append(sum(g_rewards))
        average_reward += sum(g_rewards)

        save_rover_path(final_rover_path, "Rover_Paths")

        srun += 1

    average_reward /= stat_runs
    print(average_reward)
    save_reward_history(reward_history, "Final_GlobalRewards.csv")
    if p["vis_running"]:
        run_visualizer()


if __name__ == '__main__':

    # Test Performance of Skills in Agent Skill Set
    for skill_id in range(p["n_skills"]):
        test_skill_performance(skill_id)

    # Test Performance of CBA
    rover_suggestions = np.ones(p["n_rovers"], int)
    if p["suggestion_type"] == "Identical":
        suggestion_id = 1
        rover_suggestions *= suggestion_id
    elif p["suggestion_type"] == "Unique":
        for rov_id in range(p["n_rovers"]):
            if rov_id < p["n_poi"]:
                rover_suggestions[rov_id] = rov_id
            else:
                rover_suggestions[rov_id] = random.randint(0, p["n_skills"]-1)
    elif p["suggestion_type"] == "Random":
        for rov_id in range(p["n_rovers"]):
            rover_suggestions[rov_id] = random.randint(0, p["n_skills"]-1)
    else:  # Custom
        rover_suggestions = [0, 1, 0]
    print(rover_suggestions)
    test_suggestions_policy_bank(p["policy_bank_type"], rover_suggestions)

