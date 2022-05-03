from ccea import Ccea
from RewardFunctions.reward_functions import calc_difference, calc_dpp
from RewardFunctions.cba_rewards import calc_sdpp
from RoverDomain_Core.rover_domain import RoverDomain
import pickle
import csv
import os
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


def rover_global():
    """
    Train rovers in the classic rover domain using the global reward
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    srun = p["starting_srun"]
    while srun < stat_runs:  # Perform statistical runs
        print("Run: %i" % srun)

        # Create new population of policies for each rover
        for rover_id in range(n_rovers):
            pops["EA{0}".format(rover_id)].create_new_population()
        reward_history = []

        for gen in range(generations):
            for rover_id in range(n_rovers):
                pops["EA{0}".format(rover_id)].select_policy_teams()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                # Reset rovers to initial conditions and select network weights
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    rd.rovers[rk].reset_rover()
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rd.rovers["R{0}".format(rover_id)].get_weights(weights)

                # Rover runs initial scan of environment
                for rk in rd.rovers:
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                poi_rewards = np.zeros((rd.num_pois, rover_steps))
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rovers act according to their policy
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rovers scan environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)

                    step_rewards = rd.calc_global()
                    for poi_id in range(rd.num_pois):
                        poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                # Update fitness of policies using reward information
                g_reward = 0
                for poi_id in range(rd.num_pois):
                     g_reward += max(poi_rewards[poi_id])
                for rover_id in range(n_rovers):
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    pops["EA{0}".format(rover_id)].fitness[policy_id] = g_reward

            # Testing Phase (test best policies found so far) ----------------------------------------------------------
            if gen % sample_rate == 0 or gen == generations - 1:
                # Reset rovers to initial conditions
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()

                # Rover runs initial scan of environment and selects network weights
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rd.rovers[rk].get_weights(weights)
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                poi_rewards = np.zeros((rd.num_pois, rover_steps))
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes information froms can and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)

                    step_rewards = rd.calc_global()
                    for poi_id in range(rd.num_pois):
                        poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                g_reward = 0
                for poi_id in range(rd.num_pois):
                    g_reward += max(poi_rewards[poi_id])
                reward_history.append(g_reward)
            # TEST OVER ------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                pops["EA{0}".format(rover_id)].down_select()

        save_reward_history(reward_history, "Global_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

        srun += 1


def rover_difference():
    """
    Train rovers in classic rover domain using difference rewards
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for each instance an EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    srun = p["starting_srun"]
    while srun < stat_runs:  # Perform statistical runs
        print("Run: %i" % srun)

        # Create new population of policies for each rover
        for pkey in pops:
            pops[pkey].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                # Reset rovers to initial conditions and select network weights
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    rd.rovers[rk].reset_rover()
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rd.rovers[rk].get_weights(weights)

                # Rover runs initial scan of environment
                for rk in rd.rovers:
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                d_rewards = np.zeros((n_rovers, rover_steps))
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rovers scan environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:  # Update POI observer distances
                        rd.pois[poi].update_observer_distances(rd.rovers)

                    # Calculate Rewards
                    g_reward = rd.calc_global()
                    dif_reward = calc_difference(rd.pois, g_reward)
                    for rover_id in range(n_rovers):
                        d_rewards[rover_id, step_id] = dif_reward[rover_id]

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    pops["EA{0}".format(rover_id)].fitness[policy_id] = sum(d_rewards[rover_id])

            # Testing Phase (test best policies found so far) ---------------------------------------------------------
            if gen % sample_rate == 0 or gen == generations-1:
                # Reset rovers to initial conditions
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()

                # Rover runs initial scan of environment and selects network weights
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rd.rovers[rk].get_weights(weights)
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                poi_rewards = np.zeros((rd.num_pois, rover_steps))
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes information froms can and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)

                    step_rewards = rd.calc_global()
                    for poi_id in range(rd.num_pois):
                        poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                g_reward = 0
                for poi_id in range(rd.num_pois):
                    g_reward += max(poi_rewards[poi_id])
                reward_history.append(g_reward)
            # TEST OVER ------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for pkey in pops:
                pops[pkey].down_select()

        save_reward_history(reward_history, "Difference_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

        srun += 1


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

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    srun = p["starting_srun"]
    while srun < stat_runs:  # Perform statistical runs
        print("Run: %i" % srun)

        # Create new population of policies for each rover
        for pkey in pops:
            pops[pkey].create_new_population()
        reward_history = []

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    rd.rovers[rk].reset_rover()
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rd.rovers[rk].get_weights(weights)

                # Rover runs initial scan of environment
                for rk in rd.rovers:
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                dpp_rewards = np.zeros((n_rovers, rover_steps))
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rovers scan environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)

                    g_reward = rd.calc_global()
                    rover_rewards = calc_dpp(rd.pois, g_reward)
                    for rover_id in range(n_rovers):
                        dpp_rewards[rover_id, step_id] = rover_rewards[rover_id]

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    pops["EA{0}".format(rover_id)].fitness[policy_id] = sum(dpp_rewards[rover_id])

            # Testing Phase (test best policies found so far) ----------------------------------------------------------
            if gen % sample_rate == 0 or gen == generations - 1:
                # Reset rovers to initial conditions
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()

                # Rover runs initial scan of environment and selects network weights
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rd.rovers[rk].get_weights(weights)
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                poi_rewards = np.zeros((rd.num_pois, rover_steps))
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes information froms can and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)

                    step_rewards = rd.calc_global()
                    for poi_id in range(rd.num_pois):
                        poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                g_reward = 0
                for poi_id in range(rd.num_pois):
                    g_reward += max(poi_rewards[poi_id])
                reward_history.append(g_reward)
            # TEST OVER ------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for pkey in pops:
                pops[pkey].down_select()

        save_reward_history(reward_history, "DPP_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

        srun += 1


def rover_sdpp(sgst):
    """
    Train rovers in tightly coupled rover domain using D++ with counterfactual suggestions
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    srun = p["starting_srun"]
    while srun < stat_runs:  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for pkey in pops:  # Randomly initialize ccea populations
            pops[pkey].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
                pops[pkey].reset_fitness()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    rd.rovers[rk].reset_rover()
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rd.rovers[rk].get_weights(weights)

                # Rover runs initial scan of environment
                for rk in rd.rovers:
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                sdpp_rewards = np.zeros((n_rovers, rover_steps))
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rovers scan environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)

                    g_reward = rd.calc_global()
                    rover_rewards = calc_sdpp(rd.pois, g_reward, sgst)
                    for rover_id in range(n_rovers):
                        sdpp_rewards[rover_id, step_id] = rover_rewards[rover_id]

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    pops["EA{0}".format(rover_id)].fitness[policy_id] = sum(sdpp_rewards[rover_id])

            # Testing Phase (test best policies found so far) ---------------------------------------------------------
            if gen % sample_rate == 0 or gen == generations - 1:
                # Reset rovers to initial configuration
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()
                # Rover runs initial scan of environment and uses best policy from CCEA pop
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rd.rovers[rk].get_weights(weights)

                poi_rewards = np.zeros((rd.num_pois, rover_steps))
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes information from scan and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)

                    step_rewards = rd.calc_global()
                    for poi_id in range(rd.num_pois):
                        poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                g_reward = 0
                for poi_id in range(rd.num_pois):
                    g_reward += max(poi_rewards[poi_id])
                reward_history.append(g_reward)
            # TEST OVER ------------------------------------------------------------------------------------------------

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "SDPP_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

        srun += 1


if __name__ == '__main__':
    """
    Run rover domain code
    """

    reward_type = p["reward_type"]
    sgst = [1, 1, 1, 1, 1, 1]  # Counterfactual Suggestions for S-D++ (must match number of rovers)

    if reward_type == "Global":
        print("GLOBAL REWARDS")
        rover_global()
    elif reward_type == "Difference":
        print("DIFFERENCE REWARDS")
        rover_difference()
    elif reward_type == "DPP":
        print("D++ REWARDS")
        rover_dpp()
    elif reward_type == "SDPP":
        print("S-D++ REWARDS")
        rover_sdpp(sgst)
    else:
        print("REWARD TYPE ERROR")
