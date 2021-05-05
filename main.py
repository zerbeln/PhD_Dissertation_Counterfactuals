import pyximport; pyximport.install(language_level=3)
from Python_Code.ccea import Ccea
from Python_Code.ea import EvoAlg
from Python_Code.suggestion_network import SuggestionNetwork
from Python_Code.reward_functions import calc_global_loose, calc_global_tight, calc_difference, calc_dpp, target_poi
from Python_Code.suggestion_rewards import calc_sdpp, calc_sd_reward, sdpp_and_sd
from Python_Code.rover_domain import RoverDomain
from Python_Code.agent import Rover
from Visualizer.visualizer import run_visualizer
import pickle
import math
import csv; import os
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


def save_best_policies(network_weights, srun, file_name):
    """
    Save trained neural networks as a pickle file
    """
    # Make sure Policy Bank Folder Exists
    if not os.path.exists('Policy_Bank'):  # If Data directory does not exist, create it
        os.makedirs('Policy_Bank')

    dir_name = 'Policy_Bank/SRUN{0}'.format(srun)
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    fpath_name = os.path.join(dir_name, file_name)
    weight_file = open(fpath_name, 'wb')
    pickle.dump(network_weights, weight_file)
    weight_file.close()


def load_saved_policies(file_name, srun):
    """
    Load saved Neural Network policies from pickle file
    """

    dir_name = 'Policy_Bank/SRUN{0}'.format(srun)
    fpath_name = os.path.join(dir_name, file_name)
    weight_file = open(fpath_name, 'rb')
    weights = pickle.load(weight_file)
    weight_file.close()

    return weights


def standard_rover(suggestion_type):
    """
    Run rover domain using counterfactual suggestions for D++
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    n_poi = p["n_poi"]
    reward_type = p["reward_type"]
    global_type = p["g_type"]
    rover_steps = p["steps"]

    rd = RoverDomain(new_world=False)
    rd.inital_world_setup()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(population_size)

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Create new CCEA population
        for rover_id in range(n_rovers):  # Randomly initialize ccea populations
            rovers["EA{0}".format(rover_id)].create_new_population()
        suggestion = suggestion_type
        reward_history = []

        for gen in range(generations):
            if gen % 100 == 0:
                print("Gen: %i" % gen)

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                rd.update_rover_path(rovers, -1)  # Record starting position of each rover
                for steps in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                    for rover_id in range(n_rovers):  # Rover processes scan information and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                    rd.update_rover_path(rovers, steps)

                # Update fitness of policies using reward information
                if global_type == "Loose":
                    global_reward = calc_global_loose(rd.rover_path, rd.pois)
                elif global_type == "Tight":
                    global_reward = calc_global_tight(rd.rover_path, rd.pois)
                if reward_type == "global":
                    for rover_id in range(n_rovers):
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[policy_id] = global_reward
                elif reward_type == "difference":
                    d_reward = calc_difference(rd.rover_path, rd.pois, global_reward)
                    for rover_id in range(n_rovers):
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[policy_id] = d_reward[rover_id]
                elif reward_type == "dpp":
                    dpp_reward = calc_dpp(rd.rover_path, rd.pois, global_reward)
                    for rover_id in range(n_rovers):
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[policy_id] = dpp_reward[rover_id]
                elif reward_type == "sdpp":
                    sdpp_reward = calc_sdpp(rd.rover_path, rd.pois, global_reward, suggestion)
                    for rover_id in range(n_rovers):
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[policy_id] = sdpp_reward[rover_id]

            # Testing Phase (test best policies found so far)
            for rover_id in range(n_rovers):
                rovers["Rover{0}".format(rover_id)].reset_rover()
                policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                rovers["Rover{0}".format(rover_id)].get_weights(weights)

                if gen == generations-1:
                    # Record Initial Rover Position
                    final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                    final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                    final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]

            for steps in range(rover_steps):
                for rover_id in range(n_rovers):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                for rover_id in range(n_rovers):  # Rover processes information froms can and acts
                    rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)

                    if gen == generations-1:
                        # Record Position of Each Rover
                        final_rover_path[srun, rover_id, steps + 1, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                        final_rover_path[srun, rover_id, steps + 1, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                        final_rover_path[srun, rover_id, steps + 1, 2] = rovers["Rover{0}".format(rover_id)].pos[2]

            if global_type == "Loose":
                global_reward = calc_global_loose(rd.rover_path, rd.pois)
            elif global_type == "Tight":
                global_reward = calc_global_tight(rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "SDPP_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id))

    save_rover_path(final_rover_path, "Rover_Paths")
    run_visualizer()


def train_target_poi(poi_target):
    """
    Run the rover domain using global reward for learning signal
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    n_poi = p["n_poi"]
    rover_steps = p["steps"]

    rd = RoverDomain(new_world=False)  # Number of POI, Number of Rovers
    ea = EvoAlg(population_size)
    # Create dictionary of rover instances
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Load World Configuration
        rd.inital_world_setup(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)

        # Create new EA pop
        ea.create_new_population()
        reward_history = []
        for gen in range(generations):
            # if gen % 100 == 0:
            #     print("Gen: %i" % gen)

            for pol_id in range(population_size):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    weights = ea.population["pol{0}".format(pol_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                rd.update_rover_path(rovers, -1)  # Record starting position of each rover
                for steps in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                    for rover_id in range(n_rovers):  # Rover processes scan information and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                    rd.update_rover_path(rovers, steps)

                # Update fitness of policies using reward information
                reward = target_poi(poi_target, rd.pois, rd.rover_path)
                ea.fitness[pol_id] = reward

            # Testing Phase (test best policies found so far)
            for rover_id in range(n_rovers):
                rovers["Rover{0}".format(rover_id)].reset_rover()
                policy_id = np.argmax(ea.fitness)  # Test the best policy in the evolutionary population
                weights = ea.population["pol{0}".format(policy_id)]
                rovers["Rover{0}".format(rover_id)].get_weights(weights)

            poi_observers = np.zeros((n_poi, n_rovers))
            for steps in range(rover_steps):
                for rover_id in range(n_rovers):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                for rover_id in range(n_rovers):  # Rover processes information from scan and acts
                    rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)

                    for poi_id in range(n_poi):
                        x_dist = rd.pois[poi_id, 0] - rovers["Rover{0}".format(rover_id)].pos[0]
                        y_dist = rd.pois[poi_id, 1] - rovers["Rover{0}".format(rover_id)].pos[1]
                        dist = math.sqrt((x_dist**2) + (y_dist**2))

                        if dist < rd.obs_radius:
                            poi_observers[poi_id, rover_id] = 1

            policy_accuracy = 0
            for rover_id in range(n_rovers):
                policy_accuracy += poi_observers[poi_target, rover_id]
            policy_accuracy /= n_rovers
            reward_history.append(policy_accuracy)

            ea.down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "TargetPOI{0}_Accuracy.csv".format(poi_target))
        policy_id = np.argmax(ea.fitness)
        weights = ea.population["pol{0}".format(policy_id)]
        save_best_policies(weights, srun, "GoTowardsPOI{0}".format(poi_target))


def train_behavior_selection():
    """
    Run rover domain using counterfactual suggestions for D++
    """
    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    n_suggestions = p["n_suggestions"]

    rd = RoverDomain(new_world=False)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_suggestions, n_out=n_suggestions, n_hid=8)
        rovers["SN{0}".format(rover_id)] = SuggestionNetwork(n_suggestions, n_suggestions, 8)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Load World Configuration
        rd.inital_world_setup(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population

        perf_data = []
        for gen in range(generations):
            # if gen % 100 == 0:
            #     print("Gen: %i" % gen)

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                    rovers["SN{0}".format(rover_id)].get_weights(weights)

                for sgst in range(n_suggestions):
                    suggestion = np.ones(n_suggestions) * -1
                    suggestion[sgst] = 1
                    for rover_id in range(n_rovers):
                        rovers["SN{0}".format(rover_id)].get_inputs(suggestion)
                        policy_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        rovers["Rover{0}".format(rover_id)].update_policy_belief(policy_outputs)
                        chosen_pol = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)

                        if chosen_pol == sgst:  # Network correctly selects desired policy
                            policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                            rovers["EA{0}".format(rover_id)].fitness[policy_id] += 1

            # Testing Phase (test best policies found so far)
            for rover_id in range(n_rovers):
                rovers["Rover{0}".format(rover_id)].reset_rover()
                policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                rovers["SN{0}".format(rover_id)].get_weights(weights)

            selection_accuracy = 0
            for sgst in range(n_suggestions):
                suggestion = np.ones(n_suggestions) * -1
                suggestion[sgst] = 1
                for rover_id in range(n_rovers):  # Rover scans environment
                    rovers["SN{0}".format(rover_id)].get_inputs(suggestion)
                    policy_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                    rovers["Rover{0}".format(rover_id)].update_policy_belief(policy_outputs)
                    chosen_pol = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)

                    if chosen_pol == sgst:
                        selection_accuracy += 1

            selection_accuracy /= (n_suggestions*n_rovers)
            perf_data.append(selection_accuracy)

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(perf_data, "SelectionAccuracy.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id))


def test_suggestions(sgst):
    """
    Test trained behavior selection policies
    """

    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    reward_type = p["reward_type"]
    global_type = p["g_type"]
    rover_steps = p["steps"]
    n_suggestions = p["n_suggestions"]

    rd = RoverDomain(new_world=False)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["SN{0}".format(rover_id)] = SuggestionNetwork(n_suggestions, n_suggestions, 8)

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    reward_history = []
    for srun in range(stat_runs):
        print("Run: %i" % srun)

        # Load World Configuration
        rd.inital_world_setup(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)

        # Load Pre-Trained Policies
        policy_bank = {}
        for pol_id in range(n_suggestions):
            policy_bank["Policy{0}".format(pol_id)] = load_saved_policies("GoTowardsPOI{}".format(pol_id), srun)

        for rover_id in range(n_rovers):
            weights = load_saved_policies("SelectionWeights{0}".format(rover_id), srun)
            rovers["SN{0}".format(rover_id)].get_weights(weights)

        rd.clear_rover_path()
        for rover_id in range(n_rovers):
            suggestion = np.ones(n_suggestions) * -1
            suggestion[rover_id] = 1
            rovers["Rover{0}".format(rover_id)].reset_rover()
            rovers["SN{0}".format(rover_id)].get_inputs(suggestion)
            policy_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
            rovers["Rover{0}".format(rover_id)].update_policy_belief(policy_outputs)
            chosen_pol = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
            # print("CHOSEN POLICY: ", chosen_pol)
            weights = policy_bank["Policy{0}".format(chosen_pol)]
            rovers["Rover{0}".format(rover_id)].get_weights(weights)

            # Record Initial Rover Position
            final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
            final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
            final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]

        rd.update_rover_path(rovers, -1)
        for steps in range(rover_steps):
            for rover_id in range(n_rovers):  # Rover scans environment
                suggestion = np.ones(n_suggestions) * -1
                suggestion[rover_id] = 1
                rovers["SN{0}".format(rover_id)].get_inputs(suggestion)
                policy_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                rovers["Rover{0}".format(rover_id)].update_policy_belief(policy_outputs)
                chosen_pol = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
                weights = policy_bank["Policy{0}".format(chosen_pol)]
                rovers["Rover{0}".format(rover_id)].get_weights(weights)
                rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
            for rover_id in range(n_rovers):  # Rover processes information from scan and acts
                rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)

                # Record Position of Each Rover
                final_rover_path[srun, rover_id, steps + 1, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                final_rover_path[srun, rover_id, steps + 1, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                final_rover_path[srun, rover_id, steps + 1, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
            rd.update_rover_path(rovers, steps)

        if global_type == "Loose":
            global_reward = calc_global_loose(rd.rover_path, rd.pois)
        elif global_type == "Tight":
            global_reward = calc_global_tight(rd.rover_path, rd.pois)
        reward_history.append(global_reward)

    save_reward_history(reward_history, "DBSS_Rewards.csv")
    save_rover_path(final_rover_path, "Rover_Paths")
    run_visualizer()


def create_new_world():
    """
    Create new world configuration files for POI and rovers
    """

    stat_runs = p["stat_runs"]
    rover_path = np.zeros((stat_runs, p["n_rovers"], p["steps"], 3))
    for srun in range(stat_runs):
        rd = RoverDomain(new_world=True)  # Number of POI, Number of Rovers
        rd.inital_world_setup(srun)
        rovers = {}
        for rover_id in range(p["n_rovers"]):
            rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)

        for rover_id in range(p["n_rovers"]):
            for step in range(p["steps"]):
                rover_path[srun, rover_id, step, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                rover_path[srun, rover_id, step, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                rover_path[srun, rover_id, step, 2] = rovers["Rover{0}".format(rover_id)].pos[2]

    save_rover_path(rover_path, "Rover_Paths")
    run_visualizer()


if __name__ == '__main__':
    """
    Run rover domain code
    :return:
    """

    test_type = p['test_type']
    suggestion = p["suggestion"]

    if test_type == "Create_Bank":
        # Train Behaviors Targeting Specific POI
        for poi_target in range(p["n_poi"]):
            print("TRAINING POLICY: ", poi_target)
            train_target_poi(poi_target)
    elif test_type == "Create_World":  # Create a new world configuration to train on
        create_new_world()
    elif test_type == "Train_Pol_Select":
        print("TRAINING SELECTION POLICY")
        train_behavior_selection()
    elif test_type == "Test":
        print("TESTING SELECTION POLICY")
        test_suggestions(suggestion)
    elif test_type == "Full_Train":
        for poi_target in range(p["n_poi"]):
            print("TRAINING POLICY: ", poi_target)
            train_target_poi(poi_target)
        print("TRAINING SELECTION POLICY")
        train_behavior_selection()
        print("TESTING SELECTION POLICY")
        test_suggestions(suggestion)
    else:
        standard_rover(suggestion)
