from Python_Code.ccea import Ccea
from Python_Code.suggestion_network import SuggestionNetwork
from Python_Code.rover_domain import RoverDomain
from Python_Code.agent import Rover
from Visualizer.visualizer import run_visualizer
import pickle
import csv
import os
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


if __name__ == '__main__':
    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    domain_type = p["domain_type"]
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

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs

        # Load Trained Suggestion Interpreter Weights
        for rover_id in range(n_rovers):
            s_weights = load_saved_policies('SelectionWeights{0}'.format(rover_id), rover_id, srun)
            rovers["SN{0}".format(rover_id)].get_weights(s_weights)

        # Load World Configuration
        rd.load_world(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
            final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
            final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]

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

        for rover_id in range(n_rovers):  # Initial rover scan of environment
            suggestion = s_dict["S{0}".format(test_sid[rover_id])]
            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
            sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
            sug_input = np.concatenate((suggestion, sensor_data), axis=0)
            rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
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
                final_rover_path[srun, rover_id, step_id + 1, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                final_rover_path[srun, rover_id, step_id + 1, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                final_rover_path[srun, rover_id, step_id + 1, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
            for rover_id in range(n_rovers):  # Rover scans environment
                suggestion = s_dict["S{0}".format(test_sid[rover_id])]
                rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                rovers["SN{0}".format(rover_id)].get_inputs(np.concatenate((suggestion, sensor_data), axis=0))
                rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                # rovers["Rover{0}".format(rover_id)].update_policy_belief(sug_outputs)
                # pol_id = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
                pol_id = np.argmax(sug_outputs)
                rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions.copy()

            # Calculate Global Reward
            if domain_type == "Loose":
                g_rewards[step_id] = rd.calc_global_loose()
            else:
                g_rewards[step_id] = rd.calc_global_tight()

        reward_history.append(sum(g_rewards))

        save_rover_path(final_rover_path, "Rover_Paths")

    run_visualizer()
