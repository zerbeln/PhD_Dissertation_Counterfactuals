from Python_Code.ccea import Ccea
from Python_Code.reward_functions import calc_difference_loose, calc_difference_tight, calc_dpp
from Python_Code.suggestion_rewards import calc_sdpp
from Python_Code.rover_domain import RoverDomain
from Python_Code.agent import Rover
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


def rover_global_loose():
    """
    Train rovers in the classic rover domain using the global reward
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    assert(p["coupling"] == 1)
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    rd = RoverDomain()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # World Configuration Setup
        rd.load_world(srun)
        for rover_id in range(n_rovers):  # Randomly initialize ccea populations
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                # Rover runs initial scan of environment and selects policy from CCEA pop to test
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rovers act according to their policy
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                    for rover_id in range(n_rovers):  # Rovers scan environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                    g_rewards[step_id] = rd.calc_global_loose()

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = sum(g_rewards)

            # Testing Phase (test best policies found so far)
            if gen % sample_rate == 0 or gen == generations-1:
                # Reset rovers to initial configuration and record starting position
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    if gen == generations - 1:
                        final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                        final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                        final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                # Rover runs initial scan of environment and uses best policy from CCEA pop
                for rover_id in range(n_rovers):
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes information from scan and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                        # Record Position of Rover
                        if gen == generations-1:
                            final_rover_path[srun, rover_id, step_id + 1, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                            final_rover_path[srun, rover_id, step_id + 1, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                            final_rover_path[srun, rover_id, step_id + 1, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                    for rover_id in range(n_rovers):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                    g_rewards[step_id] = rd.calc_global_loose()

                reward_history.append(sum(g_rewards))
            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()

        save_reward_history(reward_history, "Global_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

    save_rover_path(final_rover_path, "Rover_Paths")


def rover_global_tight():
    """
    Train rovers in tightly coupled rover domain using the global reward
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    assert(p["coupling"] > 1)
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    rd = RoverDomain()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # World Configuration Setup
        rd.load_world(srun)
        for rover_id in range(n_rovers):  # Randomly initialize ccea populations
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                # Rover runs initial scan of environment and selects policy from CCEA pop to test
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rovers act according to their policy
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                    for rover_id in range(n_rovers):  # Rovers scan environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                    g_rewards[step_id] = rd.calc_global_tight()

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = sum(g_rewards)

            # Testing Phase (test best policies found so far)
            if gen % sample_rate == 0 or gen == generations - 1:
                # Reset rovers to initial configuration and record starting position
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    if gen == generations - 1:
                        final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                        final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                        final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                # Rover runs initial scan of environment and uses best policy from CCEA pop
                for rover_id in range(n_rovers):
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes information from scan and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                        # Record Position of Rover
                        if gen == generations - 1:
                            final_rover_path[srun, rover_id, step_id + 1, 0] = rovers["Rover{0}".format(rover_id)].pos[
                                0]
                            final_rover_path[srun, rover_id, step_id + 1, 1] = rovers["Rover{0}".format(rover_id)].pos[
                                1]
                            final_rover_path[srun, rover_id, step_id + 1, 2] = rovers["Rover{0}".format(rover_id)].pos[
                                2]
                    for rover_id in range(n_rovers):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                    g_rewards[step_id] = rd.calc_global_tight()

                reward_history.append(sum(g_rewards))
            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()

        save_reward_history(reward_history, "Global_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

    save_rover_path(final_rover_path, "Rover_Paths")


def rover_difference_loose():
    """
    Train rovers in classic rover domain using difference rewards
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    assert (p["coupling"] == 1)
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    rd = RoverDomain()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # World Configuration Setup
        rd.load_world(srun)
        for rover_id in range(n_rovers):  # Randomly initialize ccea populations
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()
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
                    g_reward = rd.calc_global_loose()
                    dif_reward = calc_difference_loose(rd.observer_distances, rd.pois, g_reward)
                    for rover_id in range(n_rovers):
                        d_rewards[rover_id, step_id] = dif_reward[rover_id]

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = sum(d_rewards[rover_id])

            # Testing Phase (test best policies found so far)
            if gen % sample_rate == 0 or gen == generations-1:
                # Reset rovers to initial configuration and record starting position
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    if gen == generations-1:
                        final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                        final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                        final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                # Rover runs initial scan of environment and uses best policy from CCEA pop
                for rover_id in range(n_rovers):
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
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
                    g_rewards[step_id] = rd.calc_global_loose()

                reward_history.append(sum(g_rewards))

            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()

        save_reward_history(reward_history, "Difference_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

    save_rover_path(final_rover_path, "Rover_Paths")


def rover_difference_tight():
    """
    Train rovers in tightly coupled rover domain using difference rewards only
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    assert (p["coupling"] > 1)
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    rd = RoverDomain()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # World Configuration Setup
        rd.load_world(srun)
        for rover_id in range(n_rovers):  # Randomly initialize ccea populations
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()
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
                    g_reward = rd.calc_global_tight()
                    dif_reward = calc_difference_tight(rd.observer_distances, rd.pois, g_reward)
                    for rover_id in range(n_rovers):
                        d_rewards[rover_id, step_id] = dif_reward[rover_id]

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = sum(d_rewards[rover_id])

            # Testing Phase (test best policies found so far)
            if gen % sample_rate == 0 or gen == generations - 1:
                # Reset rovers to initial configuration and record starting position
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    if gen == generations - 1:
                        final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                        final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                        final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                # Rover runs initial scan of environment and uses best policy from CCEA pop
                for rover_id in range(n_rovers):
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes information froms can and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                        if gen == generations - 1:
                            # Record Position of Each Rover
                            final_rover_path[srun, rover_id, step_id + 1, 0] = rovers["Rover{0}".format(rover_id)].pos[
                                0]
                            final_rover_path[srun, rover_id, step_id + 1, 1] = rovers["Rover{0}".format(rover_id)].pos[
                                1]
                            final_rover_path[srun, rover_id, step_id + 1, 2] = rovers["Rover{0}".format(rover_id)].pos[
                                2]
                    for rover_id in range(n_rovers):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                    g_rewards[step_id] = rd.calc_global_tight()

                reward_history.append(sum(g_rewards))

            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()

        save_reward_history(reward_history, "Difference_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

    save_rover_path(final_rover_path, "Rover_Paths")


def rover_dpp():
    """
    Train rovers in tightly coupled rover domain using D++
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]
    assert (p["coupling"] > 1)

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    rd = RoverDomain()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # World Configuration Setup
        rd.load_world(srun)
        for rover_id in range(n_rovers):  # Randomly initialize ccea populations
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()
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
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = sum(dpp_rewards[rover_id])

            # Testing Phase (test best policies found so far)
            if gen % sample_rate == 0 or gen == generations - 1:
                # Reset rovers to initial configuration and record starting position
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    if gen == generations - 1:
                        # Record Initial Rover Position
                        final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                        final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                        final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                # Rover runs initial scan of environment and uses best policy from CCEA pop
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes information from scan and acts
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

                reward_history.append(sum(g_rewards))

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "DPP_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

    save_rover_path(final_rover_path, "Rover_Paths")


def rover_sdpp(sgst):
    """
    Train rovers in tightly coupled rover domain using D++
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]
    assert (p["coupling"] > 1)

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    rd = RoverDomain()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # World Configuration Setup
        rd.load_world(srun)
        for rover_id in range(n_rovers):  # Randomly initialize ccea populations
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                for rover_id in range(n_rovers):  # Rover runs initial scan of environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                sdpp_rewards = np.zeros((n_rovers, rover_steps))
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes scan information and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                    for rover_id in range(n_rovers):  # Rovers scan environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)

                    g_reward = rd.calc_global_tight()
                    rover_rewards = calc_sdpp(rd.observer_distances, rd.pois, g_reward, sgst)
                    for rover_id in range(n_rovers):
                        sdpp_rewards[rover_id, step_id] = rover_rewards[rover_id]

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = sum(sdpp_rewards[rover_id])

            # Testing Phase (test best policies found so far)
            if gen % sample_rate == 0 or gen == generations - 1:
                # Reset rovers to initial configuration and record starting position
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    if gen == generations - 1:
                        # Record Initial Rover Position
                        final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                        final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                        final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]
                # Rover runs initial scan of environment and uses best policy from CCEA pop
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes information from scan and acts
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

                reward_history.append(sum(g_rewards))

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "SDPP_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

    save_rover_path(final_rover_path, "Rover_Paths")


if __name__ == '__main__':
    """
    Run rover domain code
    """

    reward_type = p["reward_type"]
    domain_type = p["domain_type"]
    sgst = [0, 0, 0]

    if domain_type == "Loose":
        if reward_type == "Global":
            print("GLOBAL REWARDS LOOSE")
            rover_global_loose()
        elif reward_type == "Difference":
            print("DIFFERENCE REWARDS LOOSE")
            rover_difference_loose()
        else:
            print("REWARD TYPE ERROR")
    elif domain_type == "Tight":
        if reward_type == "Global":
            print("GLOBAL REWARDS TIGHT")
            rover_global_tight()
        elif reward_type == "Difference":
            print("DIFFERENCE REWARDS TIGHT")
            rover_difference_tight()
        elif reward_type == "DPP":
            print("DPP REWARDS TIGHT")
            rover_dpp()
        elif reward_type == "SDPP":
            rover_sdpp(sgst)
        else:
            print("REWARD TYPE ERROR")
    else:
        print("DOMAIN TYPE ERROR")
