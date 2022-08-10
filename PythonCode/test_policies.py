from RoverDomain_Core.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
from rover_neural_network import NeuralNetwork
import pickle
import os
import numpy as np
from parameters import parameters as p
from global_functions import create_csv_file, create_pickle_file


def load_saved_policies_python(file_name, rover_id, srun):
    """
    Load saved Neural Network policies from pickle file
    """

    dir_name = 'Policy_Bank/Rover{0}/SRUN{1}'.format(rover_id, srun)
    fpath_name = os.path.join(dir_name, file_name)
    weight_file = open(fpath_name, 'rb')
    weights = pickle.load(weight_file)
    weight_file.close()

    return weights


def test_trained_policy():
    """
    Test rover policy trained using Global, Difference, or D++ rewards.
    """
    # World Setup
    rd = RoverDomain()  # Create instance of the rover domain
    rd.load_world()

    # Generate Hazard Areas (If Testing For Hazards)
    if p["active_hazards"]:
        for poi_id in p["hazardous_poi"]:
            rd.pois["P{0}".format(poi_id)].hazardous = True

    networks = {}
    for rover_id in range(p["n_rovers"]):
        networks["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    # Data tracking
    reward_history = []  # Keep track of team performance throughout training
    incursion_tracker = []  # Keep track of the number of hazard area violations each stat run
    average_reward = 0  # Track average reward across runs
    final_rover_path = np.zeros((p["stat_runs"], p["n_rovers"], p["steps"], 3))  # Track rover trajectories

    # Run tests
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        # Load Trained Rover Networks
        for rv in rd.rovers:
            rover_id = rd.rovers[rv].rover_id
            weights = load_saved_policies_python('RoverWeights{0}'.format(rover_id), rover_id, srun)
            networks["NN{0}".format(rd.rovers[rv].rover_id)].get_weights(weights)

        n_incursions = 0
        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        rd.reset_world()
        for step_id in range(p["steps"]):
            # Get rover actions from neural network
            rover_actions = []
            for rv in rd.rovers:
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 0] = rd.rovers[rv].loc[0]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 1] = rd.rovers[rv].loc[1]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 2] = rd.rovers[rv].loc[2]

                # Get actions from rover neural networks
                rover_id = rd.rovers[rv].rover_id
                action = networks["NN{0}".format(rover_id)].run_rover_nn(rd.rovers[rv].observations)
                rover_actions.append(action)

            # Environment takes in rover actions and returns next state and global reward
            step_rewards = rd.step(rover_actions)
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]
                if rd.pois["P{0}".format(poi_id)].hazardous:
                    for dist in rd.pois["P{0}".format(poi_id)].observer_distances:
                        if dist < p["observation_radius"]:
                            n_incursions += 1

        # Calculate episodic global reward
        g_reward = 0
        for p_reward in poi_rewards:
            g_reward += max(p_reward)
        g_reward -= (n_incursions * 10)
        reward_history.append(g_reward)
        incursion_tracker.append(n_incursions)
        average_reward += g_reward
        srun += 1

    print(average_reward/p["stat_runs"])
    create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths")
    create_csv_file(reward_history, "Output_Data/", "Final_GlobalRewards.csv")
    create_csv_file(incursion_tracker, "Output_Data/", "HazardIncursions.csv")
    if p["vis_running"]:
        run_visualizer()


if __name__ == '__main__':
    test_trained_policy()
