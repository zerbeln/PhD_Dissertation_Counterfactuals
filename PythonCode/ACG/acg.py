from ACG.ea import EA
from rover_neural_network import NeuralNetwork
from RoverDomain_Core.rover_domain import RoverDomain
from ACG.supervisor import Supervisor
from ACG.supervisor_neural_network import SupervisorNetwork
from parameters import parameters as p
import numpy as np
from global_functions import *
from CBA.custom_rover_skills import get_custom_action


def train_supervisor():
    """
    Train CBA rovers using a hand-crafted set of rover skills
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
        rovers_nn["RV{0}".format(rover_id)] = NeuralNetwork(n_inp=p["cba_inp"], n_hid=p["cba_hid"], n_out=p["cba_out"])

    # Create Hazardous POI Configuration
    hazardous_poi = []
    for cf_if in range(p["acg_configurations"]):
        h_poi = np.random.randint(0, 1, p["n_poi"])
        while 0 not in h_poi or 1 not in h_poi:
            h_poi = np.random.randint(0, 2, 2)
        hazardous_poi.append(h_poi)
    create_csv_file(hazardous_poi, 'Output_Data/', "Hazard_Configuration.csv")

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
        for gen in range(p["acg_generations"]):
            # Each policy in EA is tested
            sup_ea.reset_fitness()
            for pol_id in range(p["pop_size"]):
                # Select network weights
                sup_nn.get_weights(sup_ea.population["pol{0}".format(pol_id)])

                for cf_id in range(p["acg_configurations"]):
                    # Reset environment to configuration initial conditions
                    rd.reset_world(cf_id)
                    poi_rewards = np.zeros((p["n_poi"], p["steps"]))
                    n_incursions = 0  # Number of times rovers violate a hazardous area
                    # Hazardous POI Setup
                    if p["active_hazards"]:
                        for poi_id in range(p["n_poi"]):
                            if hazardous_poi[cf_id][poi_id] == 1:
                                rd.pois["P{0}".format(poi_id)].hazardous = True
                            else:
                                rd.pois["P{0}".format(poi_id)].hazardous = False

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
                            nn_output = rovers_nn["RV{0}".format(rover_id)].run_rover_nn(rover_input)
                            chsn_pol = int(np.argmax(nn_output))
                            action = get_custom_action(chsn_pol, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                            rover_actions.append(action)

                        rd.step(rover_actions)

                        # Calculate Rewards
                        step_rewards = rd.calc_global()
                        for poi_id in range(p["n_poi"]):
                            poi_rewards[poi_id, step_id] = step_rewards[poi_id]
                            if rd.pois["P{0}".format(poi_id)].hazardous:
                                for dist in rd.pois["P{0}".format(poi_id)].observer_distances:
                                    if dist < p["observation_radius"]:
                                        n_incursions += 1

                    # Update policy fitness
                    g_reward = 0
                    for p_reward in poi_rewards:
                        g_reward += max(p_reward)
                    g_reward -= (n_incursions * 10)
                    sup_ea.fitness[pol_id] += g_reward

            # Record training data
            if gen % p["sample_rate"] == 0 or gen == p["acg_generations"]-1:
                training_rewards.append(max(sup_ea.fitness))

            # Choose parents and create new offspring population
            sup_ea.down_select()

        # Record trial data and supervisor network information
        policy_id = np.argmax(sup_ea.fitness)
        weights = sup_ea.population["pol{0}".format(policy_id)]
        save_best_policies(weights, srun, "SupervisorWeights", p["n_rovers"]+1)
        create_csv_file(training_rewards, 'Output_Data/', "ACG_Rewards.csv")

        srun += 1
