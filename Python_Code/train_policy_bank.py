from Python_Code.ccea import Ccea
from Python_Code.local_rewards import *
from Python_Code.rover_domain import RoverDomain
import pickle
import os
import numpy as np
import csv
from parameters import parameters as p
import time


def save_time_history(time_history, file_name):
    """
    Save reward data as a CSV file for graph generation. CSV is appended each time function is called.
    """

    dir_name = 'Output_Data/'  # Intended directory for output files
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, file_name)
    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(time_history)


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


def train_towards_teammates():
    """
    Train rover policy for travelling towards teammates
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["pbank_generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for pkey in pops:
            pops[pkey].create_new_population()

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()
                    rover_id = rd.rovers[rk].self_id
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rd.rovers[rk].get_weights(weights)
                for rk in rd.rovers:  # Initial rover scan of environment
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                        reward = towards_teammates_reward(rd.rovers, rd.rovers[rk].self_id)
                        rover_rewards[rd.rovers[rk].self_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    pops["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "TowardTeammates", rover_id)


def train_away_teammates():
    """
    Train rover policy for travelling away from teammates
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["pbank_generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for pkey in pops:
            pops[pkey].create_new_population()

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()
                    rover_id = rd.rovers[rk].self_id
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rd.rovers[rk].get_weights(weights)
                for rk in rd.rovers:  # Initial rover scan of environment
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                        reward = away_teammates_reward(rd.rovers, rd.rovers[rk].self_id)
                        rover_rewards[rd.rovers[rk].self_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    pops["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "AwayTeammates", rover_id)


def train_towards_poi():
    """
    Train rover policy for travelling towards POI
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["pbank_generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for pkey in pops:
            pops[pkey].create_new_population()

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    rd.rovers[rk].reset_rover()
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rd.rovers[rk].get_weights(weights)
                for rk in rd.rovers:  # Initial rover scan of environment
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    rd.update_observer_distances()
                    for rover_id in range(n_rovers):
                        reward = towards_poi_reward(rover_id, rd.observer_distances)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    pops["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "TowardPOI", rover_id)


def train_away_poi():
    """
    Train rover policy for travelling away from POI
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["pbank_generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for pkey in pops:
            pops[pkey].create_new_population()

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    rd.rovers[rk].reset_rover()
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rd.rovers[rk].get_weights(weights)
                for rk in rd.rovers:  # Initial rover scan of environment
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    rd.update_observer_distances()
                    for rover_id in range(n_rovers):
                        reward = away_poi_reward(rover_id, rd.observer_distances)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    pops["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "AwayPOI", rover_id)


def train_two_poi(target_poi):
    """
    Train rover policies for world with Two POI (left and right side of map)
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["pbank_generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for pkey in pops:
            pops[pkey].create_new_population()

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    rd.rovers[rk].reset_rover()
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rd.rovers[rk].get_weights(weights)

                # Initial rover scan of environment
                for rk in rd.rovers:
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    rd.update_observer_distances()
                    for rover_id in range(n_rovers):
                        reward = two_poi_reward(rover_id, rd.observer_distances, rd.pois, target_poi)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    pops["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "TowardPOI{0}".format(target_poi), rover_id)


def train_four_quadrants(target_q):
    """
    Train rover policies for travelling towards POI within a specific quadrant
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["pbank_generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    pops = {}
    for rover_id in range(n_rovers):
        pops["pop{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    code_run_times = np.zeros(stat_runs)
    for srun in range(stat_runs):  # Perform statistical runs
        start_time = time.time()

        # Create new CCEA populations
        for pkey in pops:
            pops[pkey].create_new_population()

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    rd.rovers[rk].reset_rover()
                    pol_id = int(pops["pop{0}".format(rover_id)].team_selection[team_id])
                    weights = pops["pop{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rd.rovers[rk].get_weights(weights)

                # Initial rover scan of environment
                for rk in rd.rovers:
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    rd.update_observer_distances()
                    for rover_id in range(n_rovers):
                        reward = four_quadrant_rewards(rover_id, rd.observer_distances, rd.pois, target_q)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(pops["pop{0}".format(rover_id)].team_selection[team_id])
                    pops["pop{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["pop{0}".format(rover_id)].fitness)
            weights = pops["pop{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "TowardQuadrant{0}".format(target_q), rover_id)

        end_time = time.time()
        code_run_times[srun] = end_time - start_time

    save_time_history(code_run_times, "RunTimes.csv")


if __name__ == '__main__':
    """
    Train policy playbooks for rover team
    """

    if p["policy_bank_type"] == "Two_POI":
        print("Training Go Towards POI: ", 0)
        train_two_poi(2)
        print("Training Go Towards POI: ", 1)
        train_two_poi(0)
    elif p["policy_bank_type"] == "Four_Quadrants":
        print("Training Go To Quadrant 0")
        train_four_quadrants(0)
        print("Training Go To Quadrant 1")
        train_four_quadrants(1)
        print("Training Go To Quadrant 2")
        train_four_quadrants(2)
        print("Training Go To Quadrant 3")
        train_four_quadrants(3)
