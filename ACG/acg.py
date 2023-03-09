from EvolutionaryAlgorithms.ea import EA
from NeuralNetworks.neural_network import NeuralNetwork
from NeuralNetworks.supervisor_neural_network import SupervisorNetwork
from RoverDomainCore.rover_domain import RoverDomain
from ACG.supervisor import Supervisor
from parameters import parameters as p
from global_functions import *
from CKI.custom_rover_skills import get_custom_action
import numpy as np


def train_supervisor_poi_hazards():
    """
    Train CKI rovers using a hand-crafted set of rover skills
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Supervisor Setup
    sup = Supervisor()
    sup_nn = SupervisorNetwork(n_inp=p["acg_inp"], n_hid=p["acg_hid"], n_out=p["acg_out"], n_agents=p["n_rovers"])
    sup_ea = EA(n_inp=p["acg_inp"], n_hid=p["acg_hid"], n_out=p["acg_out"])

    # Create rover instances
    rovers_nn = {}
    for rover_id in range(p["n_rovers"]):
        rovers_nn[f'RV{rover_id}'] = NeuralNetwork(n_inp=p["cba_inp"], n_hid=p["cba_hid"], n_out=p["cba_out"])

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        print("Run: %i" % srun)

        # Create new supervisor EA pop
        sup_ea.create_new_population()

        # Import rover neural network weights from pickle
        for rover_id in range(p["n_rovers"]):
            weights = load_saved_policies(f'RoverWeights{rover_id}', rover_id, srun)
            rovers_nn[f'RV{rover_id}'].get_weights(weights)  # CKI Network Gets Weights

        training_rewards = []
        for gen in range(p["acg_generations"]):
            # Each policy in EA is tested
            sup_ea.reset_fitness()
            for pol_id in range(p["pop_size"]):
                # Select network weights
                sup_nn.get_weights(sup_ea.population[f'pol{pol_id}'])

                for cf_id in range(p["acg_configurations"]):
                    # Reset environment to configuration initial conditions
                    rd.reset_world(cf_id)
                    poi_rewards = np.zeros((p["n_poi"], p["steps"]))
                    n_incursions = 0  # Number of times rovers violate a hazardous area

                    for step_id in range(p["steps"]):
                        # Supervisor observes environment and creates counterfactuals
                        sup.scan_environment(rd.rovers, rd.pois)
                        counterfactuals = sup_nn.run_supervisor_nn(sup.observations)

                        # Rover scans environment and processes counterfactually shaped perceptions
                        rover_actions = []
                        for rv in rd.rovers:
                            rover_id = rd.rovers[rv].rover_id
                            rd.rovers[rv].scan_environment(rd.rovers, rd.pois)
                            sensor_data = rd.rovers[rv].observations  # Unaltered sensor readings
                            c_data = counterfactuals[f'RV{rover_id}']  # Counterfactual from supervisor
                            rover_input = np.sum((sensor_data, c_data), axis=0)

                            # Run rover neural network with counterfactual information
                            nn_output = rovers_nn[f'RV{rover_id}'].run_rover_nn(rover_input)
                            chsn_pol = int(np.argmax(nn_output))
                            action = get_custom_action(chsn_pol, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                            rover_actions.append(action)

                        rd.step(rover_actions)

                        # Calculate Rewards
                        step_rewards = rd.calc_global()
                        for poi_id in range(p["n_poi"]):
                            poi_rewards[poi_id, step_id] = step_rewards[poi_id]
                            if rd.pois[f'P{poi_id}'].hazardous:
                                for dist in rd.pois[f'P{poi_id}'].observer_distances:
                                    if dist < p["observation_radius"]:
                                        n_incursions += 1

                    # Update policy fitness
                    g_reward = 0
                    for p_reward in poi_rewards:
                        g_reward += max(p_reward)
                    g_reward -= (n_incursions * p["h_penalty"])  # Penalty for rovers entering hazards
                    sup_ea.fitness[pol_id] += g_reward

                sup_ea.fitness[pol_id] /= p["acg_configurations"]  # Average reward across configurations

            # Record training data
            if gen % p["sample_rate"] == 0 or gen == p["acg_generations"]-1:
                training_rewards.append(max(sup_ea.fitness))

            # Choose parents and create new offspring population
            sup_ea.down_select()

        # Record trial data and supervisor network information
        best_policy_id = np.argmax(sup_ea.fitness)
        weights = sup_ea.population[f'pol{best_policy_id}']
        save_best_policies(weights, srun, "SupervisorWeights", p["n_rovers"])
        create_csv_file(training_rewards, 'Output_Data/', "ACG_Rewards.csv")

        srun += 1


def train_supervisor_rover_loss():
    """
    Train CKI rovers using a hand-crafted set of rover skills
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Supervisor Setup
    sup = Supervisor()
    sup_nn = SupervisorNetwork(n_inp=p["acg_inp"], n_hid=p["acg_hid"], n_out=p["acg_out"], n_agents=p["n_rovers"])
    sup_ea = EA(n_inp=p["acg_inp"], n_hid=p["acg_hid"], n_out=p["acg_out"])

    # Create rover instances
    rovers_nn = {}
    for rover_id in range(p["n_rovers"]):
        rovers_nn[f'RV{rover_id}'] = NeuralNetwork(n_inp=p["cba_inp"], n_hid=p["cba_hid"], n_out=p["cba_out"])

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        print("Run: %i" % srun)

        # Create new supervisor EA pop
        sup_ea.create_new_population()

        # Import rover neural network weights from pickle
        for rover_id in range(p["n_rovers"]):
            weights = load_saved_policies(f'RoverWeights{rover_id}', rover_id, srun)
            rovers_nn[f'RV{rover_id}'].get_weights(weights)  # CKI Network Gets Weights

        training_rewards = []
        for gen in range(p["acg_generations"]):
            # Each policy in EA is tested
            sup_ea.reset_fitness()
            for pol_id in range(p["pop_size"]):
                # Select network weights
                sup_nn.get_weights(sup_ea.population[f'pol{pol_id}'])

                for cf_id in range(p["acg_configurations"]):
                    n_lost = int(p["rover_loss"][cf_id])
                    lost_rovers = np.random.randint(0, n_lost, n_lost)

                    # Reset environment to configuration initial conditions
                    rd.reset_world(0)
                    poi_rewards = np.zeros((p["n_poi"], p["steps"]))

                    for step_id in range(p["steps"]):
                        # Supervisor observes environment and creates counterfactuals
                        sup.scan_environment(rd.rovers, rd.pois)
                        counterfactuals = sup_nn.run_supervisor_nn(sup.observations)

                        # Rover scans environment and processes counterfactually shaped perceptions
                        rover_actions = []
                        for rv in rd.rovers:
                            if rd.rovers[rv].rover_id not in lost_rovers:
                                rover_id = rd.rovers[rv].rover_id
                                rd.rovers[rv].scan_environment(rd.rovers, rd.pois)
                                sensor_data = rd.rovers[rv].observations  # Unaltered sensor readings
                                c_data = counterfactuals[f'RV{rover_id}']  # Counterfactual from supervisor
                                rover_input = np.sum((sensor_data, c_data), axis=0)

                                # Run rover neural network with counterfactual information
                                nn_output = rovers_nn[f'RV{rover_id}'].run_rover_nn(rover_input)
                                chsn_pol = int(np.argmax(nn_output))
                                action = get_custom_action(chsn_pol, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                                rover_actions.append(action)
                            else:
                                # Non-functional rovers do not move (0.5 cancels out with rover movement)
                                rover_actions.append([0.5, 0.5])

                        rd.step(rover_actions)

                        # Calculate Rewards
                        step_rewards = rd.calc_global()
                        for poi_id in range(p["n_poi"]):
                            poi_rewards[poi_id, step_id] = step_rewards[poi_id]

                    # Update policy fitness
                    g_reward = 0
                    for p_reward in poi_rewards:
                        g_reward += max(p_reward)
                    sup_ea.fitness[pol_id] += g_reward

                sup_ea.fitness[pol_id] /= p["acg_configurations"]  # Average reward across configurations

            # Record training data
            if gen % p["sample_rate"] == 0 or gen == p["acg_generations"]-1:
                training_rewards.append(max(sup_ea.fitness))

            # Choose parents and create new offspring population
            sup_ea.down_select()

        # Record trial data and supervisor network information
        best_policy_id = np.argmax(sup_ea.fitness)
        weights = sup_ea.population[f'pol{best_policy_id}']
        save_best_policies(weights, srun, "SupervisorWeights", p["n_rovers"])
        create_csv_file(training_rewards, 'Output_Data/', "ACG_Rewards.csv")

        srun += 1
