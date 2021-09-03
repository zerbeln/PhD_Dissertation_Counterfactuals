from Python_Code.ccea import Ccea
from Python_Code.suggestion_network import SuggestionNetwork
from Python_Code.suggestion_rewards import calc_sdpp, calc_sd_reward, sdpp_and_sd
from Python_Code.rover_domain import RoverDomain
from Python_Code.agent import Rover
from Visualizer.visualizer import run_visualizer
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


def get_suggestion(pois, poi_q_count):
    """
    Determine which suggestion to provide to Rovers during testing
    """

    test_suggestions = np.zeros(p["n_rovers"], int)
    if p["domain_type"] == "Loose":
        poi_id = 0
        for rover_id in range(p["n_rovers"]):
            test_suggestions[rover_id] = pois[poi_id, 3] + 1
            poi_id += 1
    else:
        q_count_copy = np.sort(poi_q_count)
        for i in range(4):
            if q_count_copy[3] == poi_q_count[i]:
                test_suggestions[0:3] = i
            elif q_count_copy[2] == poi_q_count[i]:
                test_suggestions[4:6] = i
            elif q_count_copy[1] == poi_q_count[i]:
                test_suggestions[7:9] = i

    return test_suggestions


def train_suggestions_loose():
    """
        Train suggestions
        """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    assert(p["coupling"] == 1)
    sample_rate = p["sample_rate"]

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

    # Create suggestions for rovers
    s_dict = {}
    s_dict["S0"] = np.zeros(8)  # No Suggestion
    s_dict["S1"] = np.array([1, -1, -1, -1, 1, -1, -1, -1])  # Go to Quadrant 1
    s_dict["S2"] = np.array([-1, 1, -1, -1, -1, 1, -1, -1])  # Go to Quadrant 2
    s_dict["S3"] = np.array([-1, -1, 1, -1, -1, -1, 1, -1])  # Go to Quadrant 3
    s_dict["S4"] = np.array([-1, -1, -1, 1, -1, -1, -1, 1])  # Go to Quadrant 4

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Load World Configuration
        rd.load_world(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population

        # Determine Test Suggestions
        test_sid = get_suggestion(rd.pois, rd.poi_quadrant_count)

        # Load Pre-Trained Policies
        policy_bank = {}
        for rover_id in range(n_rovers):
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

        reward_history = []  # Keep track of team performance throughout training
        for gen in range(generations):
            rover_suggestions = []
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()
                s_type = random.sample(range(n_suggestions), n_suggestions)
                rover_suggestions.append(s_type)

            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                # Get weights for suggestion interpreter
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["SN{0}".format(rover_id)].get_weights(weights)
                    rovers["EA{0}".format(rover_id)].reset_fitness()

                s_id = np.zeros(n_rovers, int)  # Identifies what suggestion each rover is using
                for sgst in range(n_suggestions):
                    for rover_id in range(n_rovers):
                        rovers["Rover{0}".format(rover_id)].reset_rover()
                        s_id[rover_id] = int(rover_suggestions[rover_id][sgst])

                    for rover_id in range(n_rovers):  # Initial rover scan of environment
                        suggestion = s_dict["S{0}".format(s_id[rover_id])]
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                        sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                        rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                        # Determine action based on sensor inputs and suggestion
                        sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        # rovers["Rover{0}".format(rover_id)].update_policy_belief(sug_outputs)
                        # pol_id = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
                        pol_id = np.argmax(sug_outputs)
                        rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                        rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions.copy()

                    rover_rewards = np.zeros((n_rovers, rover_steps))  # Keep track of reward earned by each rover at t
                    for step_id in range(rover_steps):
                        for rover_id in range(n_rovers):  # Rover processes scan information and acts
                            rovers["Rover{0}".format(rover_id)].suggestion_step(rd.world_x, rd.world_y)
                        for rover_id in range(n_rovers):  # Rover scans environment
                            suggestion = s_dict["S{0}".format(s_id[rover_id])]
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                            sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                            sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                            rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                            rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                            # Choose policy based on sensors and suggestion
                            sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                            # rovers["Rover{0}".format(rover_id)].update_policy_belief(sug_outputs)
                            # pol_id = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
                            pol_id = np.argmax(sug_outputs)
                            rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                            rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions.copy()

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

            # Testing Phase (test best policies found so far)
            if gen % sample_rate == 0 or gen == generations - 1:
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["SN{0}".format(rover_id)].get_weights(weights)

                for rover_id in range(n_rovers):  # Initial rover scan of environment
                    suggestion = s_dict["S{0}".format(test_sid[rover_id])]
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                    sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                    sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                    rovers["SN{0}".format(rover_id)].get_inputs(sug_input)
                    sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                    pol_id = np.argmax(sug_outputs)
                    # rovers["Rover{0}".format(rover_id)].update_policy_belief(sug_outputs)
                    # pol_id = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
                    rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                    rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions.copy()

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes scan information and acts
                        rovers["Rover{0}".format(rover_id)].suggestion_step(rd.world_x, rd.world_y)
                    for rover_id in range(n_rovers):  # Rover scans environment
                        suggestion = s_dict["S{0}".format(test_sid[rover_id])]
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                        rovers["SN{0}".format(rover_id)].get_inputs(np.concatenate((suggestion, sensor_data), axis=0))
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                        sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        # rovers["Rover{0}".format(rover_id)].update_policy_belief(sug_outputs)
                        # pol_id = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
                        pol_id = np.argmax(sug_outputs)
                        rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                        rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions.copy()

                    # Calculate Global Reward
                    g_rewards[step_id] = rd.calc_global_loose()

                reward_history.append(sum(g_rewards))

            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()

        save_reward_history(reward_history, "Suggestion_GlobalReward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)


def train_suggestions_tight():
    """
        Train suggestions
        """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    assert(p["coupling"] > 1)
    sample_rate = p["sample_rate"]

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

    # Create suggestions for rovers
    s_dict = {}
    s_dict["S0"] = np.zeros(8)  # No Suggestion
    s_dict["S1"] = np.array([1, -1, -1, -1, 1, -1, -1, -1])  # Go to Quadrant 1
    s_dict["S2"] = np.array([-1, 1, -1, -1, -1, 1, -1, -1])  # Go to Quadrant 2
    s_dict["S3"] = np.array([-1, -1, 1, -1, -1, -1, 1, -1])  # Go to Quadrant 3
    s_dict["S4"] = np.array([-1, -1, -1, 1, -1, -1, -1, 1])  # Go to Quadrant 4

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Load World Configuration
        rd.load_world(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population

        # Determine Test Suggestions
        test_sid = get_suggestion(rd.pois, rd.poi_quadrant_count)

        # Load Pre-Trained Policies
        policy_bank = {}
        for rover_id in range(n_rovers):
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

        reward_history = []  # Keep track of team performance throughout training
        for gen in range(generations):
            rover_suggestions = []
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()
                s_type = random.sample(range(n_suggestions), n_suggestions)
                rover_suggestions.append(s_type)

            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                # Get weights for suggestion interpreter
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["SN{0}".format(rover_id)].get_weights(weights)
                    rovers["EA{0}".format(rover_id)].reset_fitness()

                s_id = np.zeros(n_rovers, int)  # Identifies what suggestion each rover is using
                for sgst in range(n_suggestions):
                    for rover_id in range(n_rovers):
                        rovers["Rover{0}".format(rover_id)].reset_rover()
                        s_id[rover_id] = int(rover_suggestions[rover_id][sgst])

                    for rover_id in range(n_rovers):  # Initial rover scan of environment
                        suggestion = s_dict["S{0}".format(s_id[rover_id])]
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                        sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                        rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                        # Determine action based on sensor inputs and suggestion
                        sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        # rovers["Rover{0}".format(rover_id)].update_policy_belief(sug_outputs)
                        # pol_id = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
                        pol_id = np.argmax(sug_outputs)
                        rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                        rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions.copy()

                    rover_rewards = np.zeros((n_rovers, rover_steps))  # Keep track of reward earned by each rover at t
                    for step_id in range(rover_steps):
                        for rover_id in range(n_rovers):  # Rover processes scan information and acts
                            rovers["Rover{0}".format(rover_id)].suggestion_step(rd.world_x, rd.world_y)
                        for rover_id in range(n_rovers):  # Rover scans environment
                            suggestion = s_dict["S{0}".format(s_id[rover_id])]
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                            sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                            sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                            rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                            rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                            # Choose policy based on sensors and suggestion
                            sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                            # rovers["Rover{0}".format(rover_id)].update_policy_belief(sug_outputs)
                            # pol_id = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
                            pol_id = np.argmax(sug_outputs)
                            rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                            rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions.copy()

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

            # Testing Phase (test best policies found so far)
            if gen % sample_rate == 0 or gen == generations - 1:
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["SN{0}".format(rover_id)].get_weights(weights)

                for rover_id in range(n_rovers):  # Initial rover scan of environment
                    suggestion = s_dict["S{0}".format(test_sid[rover_id])]
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                    sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                    sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                    rovers["SN{0}".format(rover_id)].get_inputs(sug_input)
                    sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                    pol_id = np.argmax(sug_outputs)
                    # rovers["Rover{0}".format(rover_id)].update_policy_belief(sug_outputs)
                    # pol_id = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
                    rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                    rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions.copy()

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes scan information and acts
                        rovers["Rover{0}".format(rover_id)].suggestion_step(rd.world_x, rd.world_y)
                    for rover_id in range(n_rovers):  # Rover scans environment
                        suggestion = s_dict["S{0}".format(test_sid[rover_id])]
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                        rovers["SN{0}".format(rover_id)].get_inputs(np.concatenate((suggestion, sensor_data), axis=0))
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                        sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        # rovers["Rover{0}".format(rover_id)].update_policy_belief(sug_outputs)
                        # pol_id = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
                        pol_id = np.argmax(sug_outputs)
                        rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                        rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions.copy()

                    # Calculate Global Reward
                    g_rewards[step_id] = rd.calc_global_tight()

                reward_history.append(sum(g_rewards))

            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()

        save_reward_history(reward_history, "Suggestion_GlobalReward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)


if __name__ == '__main__':
    """
    Train suggestions
    """

    domain_type = p["domain_type"]

    if domain_type == "Loose":
        train_suggestions_loose()
    elif domain_type == "Tight":
        train_suggestions_tight()
    else:
        print("DOMAIN TYPE ERROR")
