from ACG.ea import EA
from rover_neural_network import NeuralNetwork
from RoverDomain_Core.rover_domain import RoverDomain
from ACG.supervisor import Supervisor
from ACG.supervisor_neural_network import SupervisorNetwork
from parameters import parameters as p
import numpy as np
from global_functions import *


def train_supervisor():
    """
    Train CBA rovers using a hand-crafted set of rover skills
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    if p["active_hazards"]:
        for poi_id in p["hazardous_poi"]:
            rd.pois["P{0}".format(poi_id)].hazardous = True

    # Supervisor Setup
    sup = Supervisor()
    sup_nn = SupervisorNetwork(n_agents=p["n_rovers"])
    sup_ea = EA(n_agents=p["n_rovers"])

    # Create rover instances
    rovers_nn = {}
    for rover_id in range(p["n_rovers"]):
        rovers_nn["RV{0}".format(rover_id)] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        print("Run: %i" % srun)

        # Create new supervisor EA pop
        sup_ea.create_new_population()

        # Import rover neural network weights from pickle
        for rover_id in range(p["n_rovers"]):
            weights = load_saved_policies('RoverWeights{0}'.format(rover_id), rover_id, srun)
            rovers_nn["RV{0}".format(rover_id)].get_weights(weights)  # CBA Network Gets Weights

        training_rewards = []
        for gen in range(p["generations"]):
            for pol_id in range(p["pop_size"]):
                # Each policy in EA is tested
                sup_rewards = np.zeros(p["steps"])  # Keep track of rover rewards at each t
                sup_nn.get_weights(sup_ea.population["pol{0}".format(pol_id)])

                # Reset environment to initial conditions and select network weights
                rd.reset_world()
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
                        c_data = counterfactuals["RV{0}".format(rover_id)]  # Counterfactual from supervisor
                        rover_input = np.sum((sensor_data, c_data), axis=0)

                        # Run rover neural network with counterfactual information
                        action = rovers_nn["RV{0}".format(rover_id)].run_rover_nn(rover_input)
                        rover_actions.append(action)

                    rd.step(rover_actions)

                    # Calculate Rewards
                    poi_rewards = rd.calc_global()
                    sup_rewards[step_id] = np.sum(poi_rewards)
                    for poi in rd.pois:
                        if rd.pois[poi].hazardous:
                            for dist in rd.pois[poi].observer_distances:
                                if dist < p["observation_radius"]:
                                    n_incursions += 1

                # Update policy fitness
                g_reward = np.max(sup_rewards)
                g_reward -= (n_incursions * 100)
                sup_ea.fitness[pol_id] = g_reward

            # Record training data
            if gen % p["sample_rate"] == 0:
                training_rewards.append(max(sup_ea.fitness))

            # Choose parents and create new offspring population
            sup_ea.down_select()

        # Record trial data and supervisor network information
        policy_id = np.argmax(sup_ea.fitness)
        weights = sup_ea.population["pol{0}".format(policy_id)]
        save_best_policies(weights, srun, "SupervisorWeights", p["n_rovers"]+1)
        create_csv_file(training_rewards, 'Output_Data/Rover{0}'.format(p["n_rovers"]+1), "ACG_Rewards.csv")

        srun += 1
