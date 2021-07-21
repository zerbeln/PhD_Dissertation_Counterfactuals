from Python_Code.ccea import Ccea
from Python_Code.ea import EvoAlg
from Python_Code.suggestion_network import SuggestionNetwork
from Python_Code.reward_functions import calc_difference_loose, calc_difference_tight, calc_dpp, target_poi
from Python_Code.suggestion_rewards import calc_sdpp, calc_sd_reward, sdpp_and_sd
from Python_Code.rover_domain import RoverDomain
from Python_Code.agent import Rover
from Visualizer.visualizer import run_visualizer
import pickle
import math
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


def standard_global():
    """
    Run rover domain using counterfactual suggestions for D++
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    domain_type = p["g_type"]
    rover_steps = p["steps"]

    rd = RoverDomain(new_world=False)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size)

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # World Configuration Setup
        rd.inital_world_setup(srun)
        for rover_id in range(n_rovers):  # Randomly initialize ccea populations
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                for rover_id in range(n_rovers):  # Rover runs initial scan of environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes scan information and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                    for rover_id in range(n_rovers):  # Rovers scan environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                    if domain_type == "Loose":
                        g_rewards[step_id] = rd.calc_global_loose()
                    else:
                        g_rewards[step_id] = rd.calc_global_tight()

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = max(g_rewards)

            # Testing Phase (test best policies found so far)
            if gen % 10 == 0 or gen == generations-1:
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)

                    if gen == generations-1:
                        # Record Initial Rover Position
                        final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                        final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                        final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                for rover_id in range(n_rovers):  # Rover runs initial scan of environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes information froms can and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                        if gen == generations-1:
                            # Record Position of Each Rover
                            final_rover_path[srun, rover_id, step_id + 1, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                            final_rover_path[srun, rover_id, step_id + 1, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                            final_rover_path[srun, rover_id, step_id + 1, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                    for rover_id in range(n_rovers):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                    if domain_type == "Loose":
                        g_rewards[step_id] = rd.calc_global_loose()
                    else:
                        g_rewards[step_id] = rd.calc_global_tight()

                reward_history.append(max(g_rewards))

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "Global_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

    save_rover_path(final_rover_path, "Rover_Paths")


def standard_difference():
    """
    Run rover domain using counterfactual suggestions for D++
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    domain_type = p["g_type"]
    rover_steps = p["steps"]

    rd = RoverDomain(new_world=False)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size)

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # World Configuration Setup
        rd.inital_world_setup(srun)
        for rover_id in range(n_rovers):  # Randomly initialize ccea populations
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                for rover_id in range(n_rovers):  # Rover runs initial scan of environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                d_rewards = np.zeros((n_rovers, rover_steps))
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes scan information and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                    for rover_id in range(n_rovers):  # Rovers scan environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                    if domain_type == "Loose":
                        g_reward = rd.calc_global_loose()
                        dif_reward = calc_difference_loose(rd.observer_distances, rd.pois, g_reward)
                        for rover_id in range(n_rovers):
                            d_rewards[rover_id, step_id] = dif_reward[rover_id]
                    else:
                        g_reward = rd.calc_global_tight()
                        dif_reward = calc_difference_tight(rd.observer_distances, rd.pois, g_reward)
                        for rover_id in range(n_rovers):
                            d_rewards[rover_id, step_id] = dif_reward[rover_id]

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = max(d_rewards[rover_id])

            # Testing Phase (test best policies found so far)
            if gen % 10 == 0 or gen == generations-1:
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)

                    if gen == generations-1:
                        # Record Initial Rover Position
                        final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                        final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                        final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                for rover_id in range(n_rovers):  # Rover runs initial scan of environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes information froms can and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                        if gen == generations-1:
                            # Record Position of Each Rover
                            final_rover_path[srun, rover_id, step_id + 1, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                            final_rover_path[srun, rover_id, step_id + 1, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                            final_rover_path[srun, rover_id, step_id + 1, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                    for rover_id in range(n_rovers):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                    if domain_type == "Loose":
                        g_rewards[step_id] = rd.calc_global_loose()
                    else:
                        g_rewards[step_id] = rd.calc_global_tight()

                reward_history.append(max(g_rewards))

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "Difference_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

    save_rover_path(final_rover_path, "Rover_Paths")


def standard_dpp():
    """
    Run rover domain using counterfactual suggestions for D++
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    rd = RoverDomain(new_world=False)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size)

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # World Configuration Setup
        rd.inital_world_setup(srun)
        for rover_id in range(n_rovers):  # Randomly initialize ccea populations
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                for rover_id in range(n_rovers):  # Rover runs initial scan of environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                dpp_rewards = np.zeros((n_rovers, rover_steps))
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes scan information and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                    for rover_id in range(n_rovers):  # Rovers scan environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)

                    g_reward = rd.calc_global_tight()
                    rover_rewards = calc_dpp(rd.observer_distances, rd.pois, g_reward)
                    for rover_id in range(n_rovers):
                        dpp_rewards[rover_id, step_id] = rover_rewards[rover_id]

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = max(dpp_rewards[rover_id])

            # Testing Phase (test best policies found so far)
            if gen % 10 == 0 or gen == generations - 1:
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)

                    if gen == generations - 1:
                        # Record Initial Rover Position
                        final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                        final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                        final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                for rover_id in range(n_rovers):  # Rover runs initial scan of environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes information froms can and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                        if gen == generations - 1:
                            # Record Position of Each Rover
                            final_rover_path[srun, rover_id, step_id + 1, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                            final_rover_path[srun, rover_id, step_id + 1, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                            final_rover_path[srun, rover_id, step_id + 1, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                    for rover_id in range(n_rovers):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                    g_rewards[step_id] = rd.calc_global_tight()

                reward_history.append(max(g_rewards))

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "Difference_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

    save_rover_path(final_rover_path, "Rover_Paths")


def train_target_poi(poi_target):
    """
    Run the rover domain using global reward for learning signal
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    rd = RoverDomain(new_world=False)  # Number of POI, Number of Rovers
    # Create dictionary of rover instances
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Load World Configuration
        rd.inital_world_setup(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)

        # Create new EA pop
        for rover_id in range(n_rovers):
            rovers["EA{0}".format(rover_id)].create_new_population()
        reward_history = []
        accuracy_history = []
        for gen in range(generations):
            pop_accuracy = np.zeros((n_rovers, population_size))
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    pol_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                for rover_id in range(n_rovers):  # Initial rover scan of environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes scan information and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                    for rover_id in range(n_rovers):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    reward, accuracy = target_poi(poi_target, rd.observer_distances, rd.pois, rover_id)
                    pol_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_id])
                    rovers["EA{0}".format(rover_id)].fitness[pol_id] = reward
                    pop_accuracy[rover_id, pol_id] += accuracy

            # Record accuracy of the best policy every 10th generation
            if gen % 10 == 0:
                avg_fit = 0
                avg_acc = 0
                # print("Gen: ", gen)
                for rover_id in range(n_rovers):
                    avg_fit += max(rovers["EA{0}".format(rover_id)].fitness)
                    max_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    avg_acc += pop_accuracy[rover_id, max_id]

                avg_fit /= n_rovers
                avg_acc /= n_rovers
                reward_history.append(avg_fit)
                accuracy_history.append(avg_acc)

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(accuracy_history, "TargetPOI{0}_Accuracy.csv".format(poi_target))
        save_reward_history(reward_history, "TargetPOI{0}_Rewards.csv".format(poi_target))
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "GoTowardsPOI{0}".format(poi_target), rover_id)


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
    rover_steps = p["steps"]

    rd = RoverDomain(new_world=False)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_suggestions+8, n_out=n_suggestions, n_hid=10)
        rovers["SN{0}".format(rover_id)] = SuggestionNetwork(n_suggestions+8, n_suggestions, 10)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Load World Configuration
        rd.inital_world_setup(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population

        # Load Pre-Trained Policies
        policy_bank = {}
        for rover_id in range(n_rovers):
            for pol_id in range(n_suggestions):
                policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)] = load_saved_policies("GoTowardsPOI{}".format(pol_id), rover_id, srun)

        perf_data = []
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
                    suggestion = np.ones(n_suggestions)*-1
                    suggestion[sgst] = 1

                    for rover_id in range(n_rovers):
                        rovers["Rover{0}".format(rover_id)].reset_rover()

                    for rover_id in range(n_rovers):  # Initial rover scan of environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        sensor_input = np.concatenate((suggestion, rovers["Rover{0}".format(rover_id)].sensor_readings), axis=0)
                        rovers["SN{0}".format(rover_id)].get_inputs(sensor_input)

                        # Choose policy based on sensors and suggestion
                        policy_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        chosen_pol = np.argmax(policy_outputs)
                        weights = policy_bank["Rover{0}Policy{1}".format(rover_id, chosen_pol)]
                        rovers["Rover{0}".format(rover_id)].get_weights(weights)

                    for step_id in range(rover_steps):
                        for rover_id in range(n_rovers):  # Rover processes scan information and acts
                            rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                        for rover_id in range(n_rovers):  # Rover scans environment
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                            sensor_input = np.concatenate((suggestion, rovers["Rover{0}".format(rover_id)].sensor_readings), axis=0)
                            rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                            rovers["SN{0}".format(rover_id)].get_inputs(sensor_input)

                            # Choose policy based on sensors and suggestion
                            policy_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                            chosen_pol = np.argmax(policy_outputs)
                            weights = policy_bank["Rover{0}Policy{1}".format(rover_id, chosen_pol)]
                            rovers["Rover{0}".format(rover_id)].get_weights(weights)

                            if chosen_pol == sgst:  # Network correctly selects desired policy
                                policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                                rovers["EA{0}".format(rover_id)].fitness[policy_id] += 1

                    for rover_id in range(n_rovers):
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[policy_id] /= rover_steps

            # Testing Phase (test best policies found so far)
            if gen % 10 == 0:
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["SN{0}".format(rover_id)].get_weights(weights)

                selection_accuracy = 0
                for sgst in range(n_suggestions):
                    suggestion = np.ones(n_suggestions)*-1
                    suggestion[sgst] = 1
                    for rover_id in range(n_rovers):  # Initial rover scan of environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        sensor_input = np.concatenate((suggestion, rovers["Rover{0}".format(rover_id)].sensor_readings), axis=0)
                        rovers["SN{0}".format(rover_id)].get_inputs(sensor_input)
                        policy_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        chosen_pol = np.argmax(policy_outputs)
                        weights = policy_bank["Rover{0}Policy{1}".format(rover_id, chosen_pol)]
                        rovers["Rover{0}".format(rover_id)].get_weights(weights)

                    for step_id in range(rover_steps):
                        for rover_id in range(n_rovers):  # Rover processes scan information and acts
                            rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                        for rover_id in range(n_rovers):  # Rover scans environment
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                            sensor_input = rovers["Rover{0}".format(rover_id)].sensor_readings
                            rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                            rovers["SN{0}".format(rover_id)].get_inputs(np.concatenate((suggestion, sensor_input), axis=0))
                            policy_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                            chosen_pol = np.argmax(policy_outputs)
                            weights = policy_bank["Rover{0}Policy{1}".format(rover_id, chosen_pol)]
                            rovers["Rover{0}".format(rover_id)].get_weights(weights)

                            if chosen_pol == sgst:
                                selection_accuracy += 1

                selection_accuracy /= (n_suggestions*n_rovers*rover_steps)
                perf_data.append(selection_accuracy)

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(perf_data, "SelectionAccuracy.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)


def test_suggestions(sgst):
    """
    Test trained behavior selection policies
    """

    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    domain_type = p["g_type"]
    rover_steps = p["steps"]
    n_suggestions = p["n_suggestions"]

    rd = RoverDomain(new_world=False)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["SN{0}".format(rover_id)] = SuggestionNetwork(n_suggestions+8, n_suggestions, 10)

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
        for rover_id in range(n_rovers):
            for pol_id in range(n_suggestions):
                policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)] = load_saved_policies("GoTowardsPOI{}".format(pol_id), rover_id, srun)
            weights = load_saved_policies("SelectionWeights{0}".format(rover_id), rover_id, srun)
            rovers["SN{0}".format(rover_id)].get_weights(weights)

        for rover_id in range(n_rovers):
            suggestion = np.ones(n_suggestions) * -1
            if domain_type == "Loose":
                suggestion[rover_id] = 1
            else:
                if rover_id % 2 == 0:
                    suggestion[0] = 1
                else:
                    suggestion[1] = 1
            rovers["Rover{0}".format(rover_id)].reset_rover()
            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
            sensor_input = np.concatenate((suggestion, rovers["Rover{0}".format(rover_id)].sensor_readings), axis=0)
            rovers["SN{0}".format(rover_id)].get_inputs(sensor_input)
            policy_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
            rovers["Rover{0}".format(rover_id)].update_policy_belief(policy_outputs)
            chosen_pol = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
            weights = policy_bank["Rover{0}Policy{1}".format(rover_id, chosen_pol)]
            rovers["Rover{0}".format(rover_id)].get_weights(weights)

            # Record Initial Rover Position
            final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
            final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
            final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]


        global_reward = []
        for step_id in range(rover_steps):
            for rover_id in range(n_rovers):  # Rover interprets suggestion and chooses policy
                suggestion = np.ones(n_suggestions) * -1
                if domain_type == "Loose":
                    suggestion[rover_id] = 1
                else:
                    if rover_id % 2 == 0:
                        suggestion[0] = 1
                    else:
                        suggestion[1] = 1
                sensor_input = np.concatenate((suggestion, rovers["Rover{0}".format(rover_id)].sensor_readings), axis=0)
                rovers["SN{0}".format(rover_id)].get_inputs(sensor_input)
                policy_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                rovers["Rover{0}".format(rover_id)].update_policy_belief(policy_outputs)
                chosen_pol = np.argmax(rovers["Rover{0}".format(rover_id)].policy_belief)
                weights = policy_bank["Rover{0}Policy{1}".format(rover_id, chosen_pol)]
                rovers["Rover{0}".format(rover_id)].get_weights(weights)
            for rover_id in range(n_rovers):  # Rover processes information from scan and acts
                rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                # Record Position of Each Rover
                final_rover_path[srun, rover_id, step_id + 1, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                final_rover_path[srun, rover_id, step_id + 1, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                final_rover_path[srun, rover_id, step_id + 1, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
            for rover_id in range(n_rovers):
                rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)

            if domain_type == "Loose":
                global_reward.append(rd.calc_global_loose())
            elif domain_type == "Tight":
                global_reward.append(rd.calc_global_tight())
        reward_history.append(max(global_reward))

    save_reward_history(reward_history, "DBSS_Rewards.csv")
    save_rover_path(final_rover_path, "Rover_Paths")


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

    # save_rover_path(rover_path, "Rover_Paths")
    # run_visualizer()


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
    elif test_type == "Standard":
        if p["reward_type"] == "Global":
            print("GLOBAL REWARD")
            standard_global()
        elif p["reward_type"] == "Difference":
            print("DIFFERENCE REWARDS")
            standard_difference()
        elif p["reward_type"] == "DPP":
            print("D++ REWARDS")
            standard_dpp()
    elif test_type == "Visualize":
        run_visualizer()
