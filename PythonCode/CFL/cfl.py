from parameters import parameters as p
from RoverDomainCore.rover_domain import RoverDomain
from CFL.cfl_rewards import calc_cfl_dpp, calc_cfl_difference
from NeuralNetworks.neural_network import NeuralNetwork
from global_functions import create_csv_file, save_best_policies
from EvolutionaryAlgorithms.ccea import CCEA
import numpy as np


def sample_best_team(rd, pops, networks):
    """
    Sample the performance of the team comprised of the best individuals discovered so far during the learning process
    :param rd: Instance of the rover domain
    :param pops: CCEA populations
    :param networks: Dictionary containing rover neural network instances
    :return: global reward for team of best individuals
    """

    # Rover runs initial scan of environment and selects network weights
    for rv in rd.rovers:
        policy_id = np.argmax(pops["EA{0}".format(rd.rovers[rv].rover_id)].fitness)
        weights = pops["EA{0}".format(rd.rovers[rv].rover_id)].population["pol{0}".format(policy_id)]
        networks["NN{0}".format(rd.rovers[rv].rover_id)].get_weights(weights)

    g_reward = 0
    for cf_id in range(p["n_configurations"]):
        # Reset rovers to configuration initial conditions
        rd.reset_world(cf_id)
        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        for step_id in range(p["steps"]):
            # Get rover actions from neural network
            rover_actions = []
            for rv in rd.rovers:
                rover_id = rd.rovers[rv].rover_id
                nn_output = networks["NN{0}".format(rover_id)].run_rover_nn(rd.rovers[rv].observations)
                rover_actions.append(nn_output)

            step_rewards = rd.step(rover_actions)
            # Calculate rewards at current time step
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]

        for p_reward in poi_rewards:
            g_reward += max(p_reward)

    return g_reward


def rover_cdif(counterfactuals):
    """
    Train rovers in tightly coupled rover domain using D with expert counterfactuals
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for CCEA populations
    pops = {}
    networks = {}
    for rover_id in range(p["n_rovers"]):
        pops["EA{0}".format(rover_id)] = CCEA(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])
        networks["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:  # Perform statistical runs
        print("Run: %i" % srun)
        # Create new CCEA populations
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            for pkey in pops:
                pops[pkey].select_policy_teams()
                pops[pkey].reset_fitness()

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(p["pop_size"]):
                for rv in rd.rovers:
                    policy_id = int(pops["EA{0}".format(rd.rovers[rv].rover_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rd.rovers[rv].rover_id)].population["pol{0}".format(policy_id)]
                    networks["NN{0}".format(rd.rovers[rv].rover_id)].get_weights(weights)

                for cf_id in range(p["n_configurations"]):
                    rd.reset_world(cf_id)
                    poi_rewards = np.zeros((p["n_poi"], p["steps"]))
                    for step_id in range(p["steps"]):
                        # Get rover actions from neural network
                        rover_actions = []
                        for rv in rd.rovers:
                            rover_id = rd.rovers[rv].rover_id
                            nn_output = networks["NN{0}".format(rover_id)].run_rover_nn(rd.rovers[rv].observations)
                            rover_actions.append(nn_output)

                        step_rewards = rd.step(rover_actions)
                        # Calculate rewards at current time step
                        for poi_id in range(p["n_poi"]):
                            poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                    # Update fitness of policies using reward information
                    g_reward = 0
                    for p_reward in poi_rewards:
                        g_reward += max(p_reward)
                    cfl_d_rewards = calc_cfl_difference(rd.pois, g_reward, rd.rover_poi_distances, counterfactuals)
                    for rover_id in range(p["n_rovers"]):
                        policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                        pops["EA{0}".format(rover_id)].fitness[policy_id] += cfl_d_rewards[rover_id]

            # Testing Phase (test best agent team found so far) ------------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(rd, pops, networks))
            # --------------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for pkey in pops:
                pops[pkey].down_select()

        # Record Output Files
        create_csv_file(reward_history, "Output_Data/", "CFL_D_Rewards.csv")
        for rover_id in range(p["n_rovers"]):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

        srun += 1


def rover_cdpp(counterfactuals):
    """
    Train rovers in tightly coupled rover domain using D++ with expert counterfactuals
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for CCEA populations
    pops = {}
    networks = {}
    for rover_id in range(p["n_rovers"]):
        pops["EA{0}".format(rover_id)] = Ccea(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])
        networks["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:  # Perform statistical runs
        print("Run: %i" % srun)
        # Create new CCEA populations
        for pkey in pops:
            pops[pkey].create_new_population()

        reward_history = []
        for gen in range(p["generations"]):
            for pkey in pops:
                pops[pkey].select_policy_teams()
                pops[pkey].reset_fitness()

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(p["pop_size"]):
                for rv in rd.rovers:
                    policy_id = int(pops["EA{0}".format(rd.rovers[rv].rover_id)].team_selection[team_number])
                    weights = pops["EA{0}".format(rd.rovers[rv].rover_id)].population["pol{0}".format(policy_id)]
                    networks["NN{0}".format(rd.rovers[rv].rover_id)].get_weights(weights)

                for cf_id in range(p["n_configurations"]):
                    rd.reset_world(cf_id)
                    poi_rewards = np.zeros((p["n_poi"], p["steps"]))
                    for step_id in range(p["steps"]):
                        # Get rover actions from neural network
                        rover_actions = []
                        for rv in rd.rovers:
                            rover_id = rd.rovers[rv].rover_id
                            nn_output = networks["NN{0}".format(rover_id)].run_rover_nn(rd.rovers[rv].observations)
                            rover_actions.append(nn_output)

                        step_rewards = rd.step(rover_actions)
                        # Calculate rewards at current time step
                        for poi_id in range(p["n_poi"]):
                            poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                    # Update fitness of policies using reward information
                    g_reward = 0
                    for p_reward in poi_rewards:
                        g_reward += max(p_reward)
                    cfl_dpp_rewards = calc_cfl_dpp(rd.pois, g_reward, rd.rover_poi_distances, counterfactuals)
                    for rover_id in range(p["n_rovers"]):
                        policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                        pops["EA{0}".format(rover_id)].fitness[policy_id] += cfl_dpp_rewards[rover_id]

            # Testing Phase (test best agent team found so far) ------------------------------------------------------
            if gen % p["sample_rate"] == 0 or gen == p["generations"] - 1:
                reward_history.append(sample_best_team(rd, pops, networks))
            # --------------------------------------------------------------------------------------------------------

            # Choose new parents and create new offspring population
            for pkey in pops:
                pops[pkey].down_select()

        # Record Output Files
        create_csv_file(reward_history, "Output_Data/", "CFL_DPP_Rewards.csv")
        for rover_id in range(p["n_rovers"]):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "RoverWeights{0}".format(rover_id), rover_id)

        srun += 1
