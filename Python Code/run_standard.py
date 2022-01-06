from Python_Code.ccea import Ccea
from Python_Code.reward_functions import calc_difference, calc_dpp
from Python_Code.suggestion_rewards import calc_sdpp
from Python_Code.rover_domain import RoverDomain
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

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for rover_id in range(n_rovers):  # Randomly initialize ccea populations
            rd.rovers["R{0}".format(rover_id)].reset_rover(rd.initial_rover_positions[rover_id])
            pops["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            for rover_id in range(n_rovers):
                pops["EA{0}".format(rover_id)].select_policy_teams()
                pops["EA{0}".format(rover_id)].reset_fitness()
            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()
                # Rover runs initial scan of environment and selects policy from CCEA pop to test
                for rover_id in range(n_rovers):
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rd.rovers["R{0}".format(rover_id)].get_weights(weights)
                    rd.rovers["R{0}".format(rover_id)].scan_environment(rd.rovers, rd.pois)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rovers act according to their policy
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rovers scan environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    rd.update_observer_distances()
                    g_rewards[step_id] = rd.calc_global()

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                    pops["EA{0}".format(rover_id)].fitness[policy_id] = sum(g_rewards)

            # Testing Phase (test best policies found so far) ----------------------------------------------------------
            if gen % sample_rate == 0 or gen == generations-1:
                # Reset rovers to initial configuration
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()
                # Rover runs initial scan of environment and uses best policy from CCEA pop
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rd.rovers[rk].get_weights(weights)
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes information from scan and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    rd.update_observer_distances()
                    g_rewards[step_id] = rd.calc_global()

                reward_history.append(sum(g_rewards))
            # TEST OVER ------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                pops["EA{0}".format(rover_id)].down_select()

        save_reward_history(reward_history, "Global_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)


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

    for srun in range(stat_runs):  # Perform statistical runs
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
                for rk in rd.rovers:  # Rover runs initial scan of environment
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                d_rewards = np.zeros((n_rovers, rover_steps))
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rovers scan environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    rd.update_observer_distances()
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
                # Reset rovers to initial configuration
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()
                # Rover runs initial scan of environment and uses best policy from CCEA pop
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rd.rovers[rk].get_weights(weights)
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes information froms can and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    rd.update_observer_distances()
                    g_rewards[step_id] = rd.calc_global()

                reward_history.append(sum(g_rewards))
            # TEST OVER ------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for pkey in pops:
                pops[pkey].down_select()

        save_reward_history(reward_history, "Difference_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)


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
    rd.load_world()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for pkey in pops:  # Randomly initialize ccea populations
            pops[pkey].create_new_population()  # Create new CCEA population
        reward_history = []

        for gen in range(generations):
            print(gen)
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
                for rk in rd.rovers:  # Rover runs initial scan of environment
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                dpp_rewards = np.zeros((n_rovers, rover_steps))
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rovers scan environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    rd.update_observer_distances()

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

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes information from scan and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    rd.update_observer_distances()
                    g_rewards[step_id] = rd.calc_global()

                reward_history.append(sum(g_rewards))
            # TEST OVER ------------------------------------------------------------------------------------------------

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "DPP_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)


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
    rd.load_world()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
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
                for rk in rd.rovers:  # Rover runs initial scan of environment
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                sdpp_rewards = np.zeros((n_rovers, rover_steps))
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rovers scan environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                    rd.update_observer_distances()
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

                g_rewards = np.zeros(rover_steps)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes information from scan and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                    rd.update_observer_distances()
                    g_rewards[step_id] = rd.calc_global()

                reward_history.append(sum(g_rewards))
            # TEST OVER ------------------------------------------------------------------------------------------------

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "SDPP_Reward.csv")
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)


if __name__ == '__main__':
    """
    Run rover domain code
    """

    reward_type = p["reward_type"]
    sgst = [0, 0, 0]

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
