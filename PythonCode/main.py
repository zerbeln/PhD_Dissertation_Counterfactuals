from ccea import Ccea
from suggestion_network import CBANetwork
from RoverDomain_Core.rover_domain import RoverDomain
from RewardFunctions.local_rewards import *
from global_functions import get_linear_dist, get_squared_dist, get_angle
import sys
import pickle
import csv
import os
import random
import numpy as np
from parameters import parameters as p


def save_reward_history(rover_id, reward_history, file_name):
    """
    Save reward data as a CSV file for graph generation. CSV is appended each time function is called.
    """

    dir_name = 'Output_Data/Rover{0}'.format(rover_id)  # Intended directory for output files
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, file_name)
    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(reward_history)


def save_best_policies(network_weights, srun, file_name, rover_id):
    """
    Save trained neural networks for each rover as a pickle file
    """
    # Make sure Policy Bank Folder Exists
    if not os.path.exists('Policy_Bank'):  # If Data directory does not exist, create it
        os.makedirs('Policy_Bank')

    if not os.path.exists('Policy_Bank/Rover{0}'.format(rover_id)):
        os.makedirs('Policy_Bank/Rover{0}'.format(rover_id))

    dir_name = 'Policy_Bank/Rover{0}/SRUN{1}'.format(rover_id, srun)
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    fpath_name = os.path.join(dir_name, file_name)
    weight_file = open(fpath_name, 'wb')
    pickle.dump(network_weights, weight_file)
    weight_file.close()


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
        angle = get_angle(pois[poi].x_position, pois[poi].y_position, p["x_dim"]/2, p["y_dim"]/2)
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
            angle = get_angle(rov_x, rov_y, p["x_dim"]/2, p["y_dim"]/2)
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


def train_cba_skill_selector():
    """
    Train suggestions using a pre-trained playbook of rover policies
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]

    # Suggestion Parameters
    n_skills = p["n_skills"]
    pbank_type = p["skill_type"]
    s_inp = p["s_inputs"]
    s_hid = p["s_hidden"]
    s_out = p["s_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=s_inp, n_out=s_out, n_hid=s_hid)
        pops["CBA{0}".format(rover_id)] = CBANetwork()

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < stat_runs:
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop and Load Pre-Trained Policies
        for rover_id in range(n_rovers):
            pops["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
            rd.rovers["R{0}".format(rover_id)].policy_bank = create_policy_bank(pbank_type, rover_id, srun)

        policy_rewards = [[] for i in range(n_rovers)]
        for gen in range(generations):
            # Create list of suggestions for rovers to use during training and reset rovers to initial positions
            rover_skills = []
            for rover_id in range(n_rovers):
                pops["EA{0}".format(rover_id)].select_policy_teams()
                pops["EA{0}".format(rover_id)].reset_fitness()
                skill_sample = random.sample(range(n_skills), n_skills)
                rover_skills.append(skill_sample)

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(population_size):
                rover_rewards = np.zeros((n_rovers, rover_steps))  # Keep track of rover rewards at each t
                for skill in range(n_skills):
                    # Get weights for CBA skill selector
                    for rover_id in range(n_rovers):
                        policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                        weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                        pops["CBA{0}".format(rover_id)].get_weights(weights)  # Suggestion Network Gets Weights

                    # Reset rovers to initial conditions
                    for rov in rd.rovers:
                        rd.rovers[rov].reset_rover()

                    chosen_pol = np.zeros(n_rovers)
                    for rov in rd.rovers:  # Initial rover scan of environment
                        rover_id = rd.rovers[rov].self_id
                        rd.rovers[rov].scan_environment(rd.rovers, rd.pois)
                        sensor_data = rd.rovers[rov].sensor_readings  # Unaltered sensor readings

                        # Select a skill using counterfactually shaped state information
                        target_pid = int(rover_skills[rover_id][skill])
                        suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, target_pid)
                        cba_input = np.sum((suggestion, sensor_data), axis=0)  # Shaped agent perception
                        pops["CBA{0}".format(rover_id)].get_inputs(cba_input)  # CBA network receives shaped input
                        cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()  # CBA picks skill
                        chosen_pol[rover_id] = int(cba_outputs)

                        # Rover uses selected skill
                        weights = rd.rovers[rov].policy_bank["Policy{0}".format(int(chosen_pol[rover_id]))]
                        rd.rovers[rov].get_weights(weights)
                        rd.rovers[rov].get_nn_outputs()

                    for step_id in range(rover_steps):
                        # Rover takes an action in the world
                        for rov in rd.rovers:
                            rd.rovers[rov].step(rd.world_x, rd.world_y)

                        # Rover scans environment and processes counterfactually shaped perceptions
                        for rov in rd.rovers:
                            rover_id = rd.rovers[rov].self_id
                            rd.rovers[rov].scan_environment(rd.rovers, rd.pois)
                            sensor_data = rd.rovers[rov].sensor_readings

                            # Select a skill using counterfactually shaped state information
                            target_pid = int(rover_skills[rover_id][skill])
                            suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, target_pid)
                            cba_input = np.sum((suggestion, sensor_data), axis=0)  # Shaped agent perception
                            pops["CBA{0}".format(rover_id)].get_inputs(cba_input)  # CBA network receives shaped input
                            cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()  # CBA picks skill
                            chosen_pol[rover_id] = int(cba_outputs)

                            # Rover uses selected skill
                            weights = rd.rovers[rov].policy_bank["Policy{0}".format(int(chosen_pol[rover_id]))]
                            rd.rovers[rov].get_weights(weights)
                            rd.rovers[rov].get_nn_outputs()

                        for poi in rd.pois:
                            rd.pois[poi].update_observer_distances(rd.rovers)

                        # Calculate Rewards
                        for rover_id in range(n_rovers):
                            reward = target_poi_reward(rover_id, rd.pois, int(rover_skills[rover_id][skill]))
                            rover_rewards[rover_id, step_id] = reward

                    # Update policy fitnesses
                    for rover_id in range(n_rovers):
                        policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                        pops["EA{0}".format(rover_id)].fitness[policy_id] += sum(rover_rewards[rover_id])/rover_steps

            # Choose parents and create new offspring population
            for rover_id in range(n_rovers):
                pops["EA{0}".format(rover_id)].down_select()

                # Record training performance data
                if gen % sample_rate == 0:
                    policy_rewards[rover_id].append(max(pops["EA{0}".format(rover_id)].fitness))

        # Record trial data
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)
            save_reward_history(rover_id, policy_rewards[rover_id], "CBA_Rewards.csv")

        srun += 1


if __name__ == '__main__':
    """
    Train suggestions interpreter (must have already pre-trained agent playbook)
    """

    print("Training CBA Skill Selector")
    train_cba_skill_selector()
