from Python_Code.ccea import Ccea
from Python_Code.local_rewards import *
from Python_Code.rover_domain import RoverDomain
from Python_Code.agent import Rover
import pickle
import os
import numpy as np
import csv
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
        writer.writerow(['Performance'] + reward_history)


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
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].reset_rover()
            rovers["EA{0}".format(rover_id)].create_new_population()

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
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
                        reward = towards_teammates_reward(rovers, rover_id)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_id])
                    rovers["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
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
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].reset_rover()
            rovers["EA{0}".format(rover_id)].create_new_population()

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
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
                        reward = away_teammates_reward(rovers, rover_id)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_id])
                    rovers["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
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
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].reset_rover()
            rovers["EA{0}".format(rover_id)].create_new_population()

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
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
                        reward = towards_poi_reward(rover_id, rd.observer_distances)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_id])
                    rovers["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
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
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].reset_rover()
            rovers["EA{0}".format(rover_id)].create_new_population()

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
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
                        reward = away_poi_reward(rover_id, rd.observer_distances)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_id])
                    rovers["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
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
    sample_rate = p["sample_rate"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].reset_rover()
            rovers["EA{0}".format(rover_id)].create_new_population()

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    pol_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)

                # Initial rover scan of environment
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes scan information and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                    for rover_id in range(n_rovers):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                        reward = two_poi_reward(rover_id, rd.observer_distances, rd.pois, target_poi)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_id])
                    rovers["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
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
    sample_rate = p["sample_rate"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].reset_rover()
            rovers["EA{0}".format(rover_id)].create_new_population()

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    pol_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)

                # Initial rover scan of environment
                for rover_id in range(n_rovers):
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)

                for step_id in range(rover_steps):
                    for rover_id in range(n_rovers):  # Rover processes scan information and acts
                        rovers["Rover{0}".format(rover_id)].step(rd.world_x, rd.world_y)
                    for rover_id in range(n_rovers):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                        reward = four_quadrant_rewards(rover_id, rd.observer_distances, rd.pois, target_q)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_id])
                    rovers["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "TowardQuadrant{0}".format(target_q), rover_id)


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