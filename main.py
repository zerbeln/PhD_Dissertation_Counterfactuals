from Python_Code.ccea import Ccea
from Python_Code.suggestion_network import SuggestionNetwork
from Python_Code.suggestion_rewards import calc_sdpp, calc_sd_reward, sdpp_and_sd
from Python_Code.rover_domain import RoverDomain
from Python_Code.agent import Rover
import math
import sys
import pickle
import csv
import os
import random
import numpy as np
from parameters import parameters as p


def save_reward_history(reward_history, file_name):
    """
    Save the reward history for the agents throughout the learning process (reward from best policy team each gen)
    """

    dir_name = 'Output_Data/'  # Intended directory for output files
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, file_name)
    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


def save_rover_path(rover_path, file_name):  # Save path rovers take using best policy found
    """
    Records the path each rover takes using best policy from CCEA (used by visualizer)
    :param rover_path:  trajectory tracker
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    rpath_name = os.path.join(dir_name, file_name)
    rover_file = open(rpath_name, 'wb')
    pickle.dump(rover_path, rover_file)
    rover_file.close()


def save_best_policies(network_weights, srun, file_name, rover_id):
    """
    Save trained neural networks as a pickle file
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


def create_policy_playbook(playbook_type, srun, n_inp, n_out, n_hid):
    """
    Choose which playbook of policies to load for the rovers
    """
    policy_bank = {}

    if playbook_type == "Four_Quadrant":
        for rover_id in range(p["n_rovers"]):
            w0 = load_saved_policies("TowardTeammates", rover_id, srun)
            w1 = load_saved_policies("AwayTeammates", rover_id, srun)
            w2 = load_saved_policies("TowardPOI", rover_id, srun)
            w3 = load_saved_policies("AwayPOI", rover_id, srun)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 0)] = SuggestionNetwork(n_inp, n_out, n_hid)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 0)].get_weights(w0)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 1)] = SuggestionNetwork(n_inp, n_out, n_hid)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 1)].get_weights(w1)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 2)] = SuggestionNetwork(n_inp, n_out, n_hid)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 2)].get_weights(w2)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 3)] = SuggestionNetwork(n_inp, n_out, n_hid)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 3)].get_weights(w3)
    elif playbook_type == "Two_POI":
        for rover_id in range(p["n_rovers"]):
            w0 = load_saved_policies("TowardPOI0", rover_id, srun)
            w1 = load_saved_policies("TowardPOI1", rover_id, srun)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 0)] = SuggestionNetwork(n_inp, n_out, n_hid)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 0)].get_weights(w0)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 1)] = SuggestionNetwork(n_inp, n_out, n_hid)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 1)].get_weights(w1)

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


def create_counterfactual_poi_state(poi_info, rover_pos, suggestion):
    """
    Construct a counterfactual state input for POI detections
    :return: Portion of the counterfactual state constructed from POI scanner
    """
    poi_state = np.zeros(int(360.0 / p["angle_res"]))
    temp_poi_dist_list = [[] for _ in range(int(360.0 / p["angle_res"]))]
    poi_quadrants = np.zeros(p["n_poi"], int)

    # Log POI distances into brackets
    n_poi = len(poi_info)
    for poi_id in range(n_poi):
        angle, dist = get_angle_dist(rover_pos[0], rover_pos[1], poi_info[poi_id, 0], poi_info[poi_id, 1])

        bracket = int(angle / p["angle_res"])
        if bracket > 3:
            bracket -= 4
        poi_quadrants[poi_id] = bracket
        if poi_info[poi_id, 3] == suggestion:
            temp_poi_dist_list[bracket].append(10*poi_info[poi_id, 2] / dist)

    # Encode POI information into the state vector
    for bracket in range(int(360 / p["angle_res"])):
        num_poi_bracket = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
        if num_poi_bracket > 0:
            if p["sensor_model"] == 'density':
                poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi_bracket  # Density Sensor
            elif p["sensor_model"] == 'summed':
                poi_state[bracket] = sum(temp_poi_dist_list[bracket])  # Summed Distance Sensor
            else:
                sys.exit('Incorrect sensor model')
        else:
            poi_state[bracket] = -1.0

    return poi_state, poi_quadrants


def construct_counterfactual_state(poi_info, rover_info, rover_id, suggestion):
    """
    Create a counteractual state input to represent agent suggestions
    """
    rover_pos = rover_info["Rover{0}".format(rover_id)].pos
    cfact_poi, poi_quadrants = create_counterfactual_poi_state(poi_info, rover_pos, suggestion)
    cfact_rover = create_counterfactual_rover_state(rover_info, rover_pos, rover_id, poi_quadrants, suggestion)

    counterfactual_state = np.zeros(8)
    for i in range(4):
        counterfactual_state[i] = cfact_poi[i]
        counterfactual_state[4 + i] = cfact_rover[i]

    return counterfactual_state


def create_counterfactual_rover_state(rover_info, rover_pos, self_id, poi_quadrants, suggestion):
    """
    Construct a counterfactual state input for rover detections
    :return: Portion of the counterfactual state vector created from rover scanner
    """
    rover_state = np.zeros(int(360.0 / p["angle_res"]))
    temp_rover_dist_list = [[] for _ in range(int(360.0 / p["angle_res"]))]

    # Log rover distances into brackets
    for rover_id in range(p["n_rovers"]):
        if self_id != rover_id:  # Ignore self
            rov_x = rover_info["Rover{0}".format(rover_id)].pos[0]
            rov_y = rover_info["Rover{0}".format(rover_id)].pos[1]

            angle, dist = get_angle_dist(rover_pos[0], rover_pos[1], rov_x, rov_y)
            bracket = int(angle / p["angle_res"])
            if bracket > 3:
                bracket -= 4
            # temp_rover_dist_list[bracket].append(1 / dist)
            if suggestion == 0 and bracket == poi_quadrants[0]:
                temp_rover_dist_list[bracket].append(10 / dist)
            elif suggestion == 1 and bracket == poi_quadrants[1]:
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


def train_suggestions_loose_direct():
    """
    Train suggestions in the loosely coupled rover domain
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    assert(p["coupling"] == 1)

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # Suggestion Parameters
    n_suggestions = p["n_suggestions"]
    s_inp = p["s_inputs"]
    s_hid = p["s_hidden"]
    s_out = p["s_outputs"]

    rd = RoverDomain()  # Create instance of the rover domain

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=s_inp, n_out=s_out, n_hid=s_hid)
        rovers["SN{0}".format(rover_id)] = SuggestionNetwork(s_inp, s_out, s_hid)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Load World Configuration
        rd.load_world(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population

        # Create list of suggestions for rovers to use during training
        rover_suggestions = []
        for rover_id in range(n_rovers):
            if rover_id % 2 == 0:
                s_type = [1, 0]
            else:
                s_type = [0, 1]
            rover_suggestions.append(s_type)

        s_id = np.zeros(n_rovers, int)  # Identifies what suggestion each rover is using
        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()

            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                # Get weights for suggestion interpreter
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["SN{0}".format(rover_id)].get_weights(weights)

                for sgst in range(n_suggestions):
                    for rover_id in range(n_rovers):
                        rovers["Rover{0}".format(rover_id)].reset_rover()
                        s_id[rover_id] = int(rover_suggestions[rover_id][sgst])

                    for rover_id in range(n_rovers):  # Initial rover scan of environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, s_id[rover_id])
                        sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                        sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                        rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                        # Determine action based on sensor inputs and suggestion
                        sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        rovers["Rover{0}".format(rover_id)].rover_actions = sug_outputs.copy()

                    rover_rewards = np.zeros((n_rovers, rover_steps))  # Keep track of reward earned by each rover at t
                    for step_id in range(rover_steps):
                        # Rover takes an action in the world
                        for rover_id in range(n_rovers):
                            rovers["Rover{0}".format(rover_id)].suggestion_step(rd.world_x, rd.world_y)

                        # Rover scans environment and processes suggestions
                        for rover_id in range(n_rovers):
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                            rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                            suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, s_id[rover_id])
                            sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                            sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                            rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                            # Choose policy based on sensors and suggestion
                            sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                            rovers["Rover{0}".format(rover_id)].rover_actions = sug_outputs.copy()

                        g_reward = rd.calc_global_loose()
                        dif_reward = calc_sd_reward(rd.observer_distances, rd.pois, g_reward, s_id)
                        for rover_id in range(n_rovers):
                            rover_rewards[rover_id, step_id] = dif_reward[rover_id]

                    for rover_id in range(n_rovers):
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[policy_id] += sum(rover_rewards[rover_id])

                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] /= n_suggestions

            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()

        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)


def train_suggestions_loose_playbook():
    """
    Train suggestions in the loosely coupled rover domain
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    assert (p["coupling"] == 1)
    pbank_type = p["policy_bank_type"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # Suggestion Parameters
    n_suggestions = p["n_suggestions"]
    s_inp = p["s_inputs"]
    s_hid = p["s_hidden"]
    s_out = p["s_outputs"]

    rd = RoverDomain()  # Create instance of the rover domain

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=s_inp, n_out=s_out, n_hid=s_hid)
        rovers["SN{0}".format(rover_id)] = SuggestionNetwork(s_inp, s_out, s_hid)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Load World Configuration
        rd.load_world(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population

        # Create list of suggestions for rovers to use during training
        rover_suggestions = []
        for rover_id in range(n_rovers):
            if rover_id % 2 == 0:
                s_type = [1, 0]
            else:
                s_type = [0, 1]
            rover_suggestions.append(s_type)

        # Load Pre-Trained Policies
        policy_bank = create_policy_playbook(pbank_type, srun, n_inp, n_out, n_hid)
        s_id = np.zeros(n_rovers, int)  # Identifies what suggestion each rover is using
        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()

            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                # Get weights for suggestion interpreter
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["SN{0}".format(rover_id)].get_weights(weights)

                for sgst in range(n_suggestions):
                    for rover_id in range(n_rovers):
                        rovers["Rover{0}".format(rover_id)].reset_rover()
                        s_id[rover_id] = int(rover_suggestions[rover_id][sgst])

                    rover_rewards = np.zeros((n_rovers, rover_steps + 1))  # Keep track of rover rewards at each t
                    for rover_id in range(n_rovers):  # Initial rover scan of environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, s_id[rover_id])
                        sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                        sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                        rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                        # Determine action based on sensor inputs and suggestion
                        sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        pol_id = np.argmax(sug_outputs)
                        if pol_id == s_id[rover_id]:
                            rover_rewards[rover_id, 0] += 1
                        rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                        rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions

                    for step_id in range(rover_steps):
                        # Rover takes an action in the world
                        for rover_id in range(n_rovers):
                            rovers["Rover{0}".format(rover_id)].suggestion_step(rd.world_x, rd.world_y)

                        # Rover scans environment and processes suggestions
                        for rover_id in range(n_rovers):
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                            rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                            suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, s_id[rover_id])
                            sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                            sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                            rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                            # Choose policy based on sensors and suggestion
                            sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                            pol_id = np.argmax(sug_outputs)
                            if pol_id == s_id[rover_id]:
                                rover_rewards[rover_id, step_id+1] += 1
                            rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                            rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions

                    for rover_id in range(n_rovers):
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[policy_id] += sum(rover_rewards[rover_id])

                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] /= n_suggestions

            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()

        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)


def train_suggestions_tight_direct():
    """
    Train suggestions in the tightly coupled rover domain
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    assert(p["coupling"] > 1)

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # Suggestion Parameters
    n_suggestions = p["n_suggestions"]
    s_inp = p["s_inputs"]
    s_hid = p["s_hidden"]
    s_out = p["s_outputs"]

    rd = RoverDomain()  # Create instance of the rover domain

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=s_inp, n_out=s_out, n_hid=s_hid)
        rovers["SN{0}".format(rover_id)] = SuggestionNetwork(s_inp, s_out, s_hid)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Load World Configuration
        rd.load_world(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population

        s_id = np.zeros(n_rovers, int)  # Identifies what suggestion each rover is using
        # Create list of suggestions for rovers to use during training
        rover_suggestions = []
        for rover_id in range(n_rovers):
            s_type = [0, 1]
            rover_suggestions.append(s_type)

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()

            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                # Get weights for suggestion interpreter
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["SN{0}".format(rover_id)].get_weights(weights)

                for sgst in range(n_suggestions):
                    for rover_id in range(n_rovers):
                        rovers["Rover{0}".format(rover_id)].reset_rover()
                        s_id[rover_id] = int(rover_suggestions[rover_id][sgst])

                    for rover_id in range(n_rovers):  # Initial rover scan of environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, s_id[rover_id])
                        sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                        sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                        rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                        # Determine action based on sensor inputs and suggestion
                        sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        rovers["Rover{0}".format(rover_id)].rover_actions = sug_outputs.copy()

                    rover_rewards = np.zeros((n_rovers, rover_steps))  # Keep track of reward earned by each rover at t
                    for step_id in range(rover_steps):
                        # Rover takes an action in the world
                        for rover_id in range(n_rovers):
                            rovers["Rover{0}".format(rover_id)].suggestion_step(rd.world_x, rd.world_y)

                        # Rover scans environment and processes suggestions
                        for rover_id in range(n_rovers):
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                            rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                            suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, s_id[rover_id])
                            sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                            sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                            rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                            # Choose policy based on sensors and suggestion
                            sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                            rovers["Rover{0}".format(rover_id)].rover_actions = sug_outputs.copy()

                        g_reward = rd.calc_global_tight()
                        dpp_reward = calc_sdpp(rd.observer_distances, rd.pois, g_reward, s_id)
                        for rover_id in range(n_rovers):
                            rover_rewards[rover_id, step_id] = dpp_reward[rover_id]

                    for rover_id in range(n_rovers):
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[policy_id] += sum(rover_rewards[rover_id])

                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] /= n_suggestions

            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()

        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)


def train_suggestions_tight_playbook():
    """
    Train suggestions in the tightly coupled rover domain using policy playbooks
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    assert (p["coupling"] > 1)
    pbank_type = p["policy_bank_type"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # Suggestion Parameters
    n_suggestions = p["n_suggestions"]
    s_inp = p["s_inputs"]
    s_hid = p["s_hidden"]
    s_out = p["s_outputs"]

    rd = RoverDomain()  # Create instance of the rover domain

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=s_inp, n_out=s_out, n_hid=s_hid)
        rovers["SN{0}".format(rover_id)] = SuggestionNetwork(s_inp, s_out, s_hid)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Load World Configuration
        rd.load_world(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population

        # Create list of suggestions for rovers to use during training
        rover_suggestions = []
        for rover_id in range(n_rovers):
            s_type = [0, 1]
            rover_suggestions.append(s_type)

        # Load Pre-Trained Policies
        policy_bank = create_policy_playbook(pbank_type, srun, n_inp, n_out, n_hid)
        s_id = np.zeros(n_rovers, int)  # Identifies what suggestion each rover is using
        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()

            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                # Get weights for suggestion interpreter
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["SN{0}".format(rover_id)].get_weights(weights)

                for sgst in range(n_suggestions):
                    for rover_id in range(n_rovers):
                        rovers["Rover{0}".format(rover_id)].reset_rover()
                        s_id[rover_id] = int(rover_suggestions[rover_id][sgst])

                    rover_rewards = np.zeros((n_rovers, rover_steps + 1))  # Keep track of rover rewards at each t
                    for rover_id in range(n_rovers):  # Initial rover scan of environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, s_id[rover_id])
                        sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                        sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                        rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                        # Determine action based on sensor inputs and suggestion
                        sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        pol_id = np.argmax(sug_outputs)
                        if pol_id == s_id[rover_id]:
                            rover_rewards[rover_id, 0] += 1
                        rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                        rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions

                    for step_id in range(rover_steps):
                        # Rover takes an action in the world
                        for rover_id in range(n_rovers):
                            rovers["Rover{0}".format(rover_id)].suggestion_step(rd.world_x, rd.world_y)

                        # Rover scans environment and processes suggestions
                        for rover_id in range(n_rovers):
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                            rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                            suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, s_id[rover_id])
                            sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                            sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                            rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                            # Choose policy based on sensors and suggestion
                            sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                            pol_id = np.argmax(sug_outputs)
                            if pol_id == s_id[rover_id]:
                                rover_rewards[rover_id, step_id+1] += 1
                            rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                            rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions

                    for rover_id in range(n_rovers):
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[policy_id] += sum(rover_rewards[rover_id])

                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] /= n_suggestions

            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()

        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)


if __name__ == '__main__':
    """
    Train suggestions
    """

    domain_type = p["domain_type"]
    pbank_type = p["policy_bank_type"]

    print("Training Suggestion Interpreter")
    if domain_type == "Loose" and pbank_type == "None":
        train_suggestions_loose_direct()
    elif domain_type == "Loose" and pbank_type != "None":
        train_suggestions_loose_playbook()
    elif domain_type == "Tight" and pbank_type == "None":
        train_suggestions_tight_direct()
    elif domain_type == "Tight" and pbank_type != "None":
        train_suggestions_tight_playbook()
    else:
        print("DOMAIN TYPE ERROR")
