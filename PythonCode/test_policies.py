from RoverDomain_Core.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
import pickle
import csv
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


def load_saved_policies_cpp(file_name, rover_id, srun):
    dir_name = 'Policy_Bank/'.format(rover_id)
    fpath_name = os.path.join(dir_name, file_name)

    # Read Weights from txt file generated from C++
    weights = []
    with open(fpath_name) as txtfile:
        fileread = csv.reader(txtfile, delimiter=',')

        for row in fileread:
            weights.append(row)

    # Convert string to float
    nn_weights = []
    for row in weights:
        row.remove(row[len(row)-1])
        for w in row:
            nn_weights.append(float(w))

    # Convert weights to dictionary form used in Python codebase
    rover_policy = {}
    w_count = 0
    layer1 = np.zeros(p["n_inputs"] * p["n_hidden"])
    layer2 = np.zeros(p["n_hidden"] * p["n_outputs"])
    b1 = np.zeros(p["n_hidden"])
    b2 = np.zeros(p["n_outputs"])
    for w in range(p["n_inputs"] * p["n_hidden"]):
        layer1[w] = nn_weights[w_count]
        w_count += 1
    for w in range(p["n_hidden"] * p["n_outputs"]):
        layer2[w] = nn_weights[w_count]
        w_count += 1
    for w in range(p["n_hidden"]):
        b1[w] = nn_weights[w_count]
        w_count += 1
    for w in range(p["n_outputs"]):
        b2[w] = nn_weights[w_count]
        w_count += 1

    rover_policy["L1"] = layer1
    rover_policy["L2"] = layer2
    rover_policy["b1"] = b1
    rover_policy["b2"] = b2

    return rover_policy


def test_trained_policy():
    """
    Test rover policy trained using Global, Difference, or D++ rewards.
    """
    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # World Setup
    rd = RoverDomain()  # Create instance of the rover domain
    rd.load_world()

    # Generate Hazard Areas (If Testing For Hazards)
    if p["active_hazards"]:
        for poi_id in p["hazardous_poi"]:
            rd.pois["P{0}".format(poi_id)].hazardous = True

    reward_history = []  # Keep track of team performance throughout training
    incursion_tracker = []  # Keep track of the number of hazard area violations each stat run
    average_reward = 0
    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    srun = p["starting_srun"]
    while srun < stat_runs:

        # Load Trained Rover Networks
        for rk in rd.rovers:
            rover_id = rd.rovers[rk].self_id
            rov_weights = load_saved_policies_python('RoverWeights{0}'.format(rover_id), rover_id, srun)
            # rov_weights = load_saved_policies_cpp('RoverPolicy{0}.txt'.format(rover_id), rover_id, srun)
            rd.rovers[rk].get_weights(rov_weights)

        # Reset Rover
        for rk in rd.rovers:
            rd.rovers[rk].reset_rover()
            final_rover_path[srun, rd.rovers[rk].self_id, 0, 0] = rd.rovers[rk].x_pos
            final_rover_path[srun, rd.rovers[rk].self_id, 0, 1] = rd.rovers[rk].y_pos
            final_rover_path[srun, rd.rovers[rk].self_id, 0, 2] = rd.rovers[rk].theta_pos

        # Initial rover scan of environment
        for rk in rd.rovers:
            rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
        for poi in rd.pois:
            rd.pois[poi].update_observer_distances(rd.rovers)

        rewards = np.zeros(p["n_poi"])
        n_incursions = 0
        for step_id in range(rover_steps):
            # Rover takes an action in the world
            for rk in rd.rovers:
                rd.rovers[rk].step(rd.world_x, rd.world_y)
                final_rover_path[srun, rd.rovers[rk].self_id, step_id + 1, 0] = rd.rovers[rk].x_pos
                final_rover_path[srun, rd.rovers[rk].self_id, step_id + 1, 1] = rd.rovers[rk].y_pos
                final_rover_path[srun, rd.rovers[rk].self_id, step_id + 1, 2] = rd.rovers[rk].theta_pos

            # Rover scans environment and processes suggestions
            for rk in rd.rovers:
                rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
            for poi in rd.pois:
                rd.pois[poi].update_observer_distances(rd.rovers)

            # Calculate Global Reward
            poi_rewards = rd.calc_global()
            for poi_id in range(p["n_poi"]):
                if rd.pois["P{0}".format(poi_id)].hazardous and poi_rewards[poi_id] < 0:
                    n_incursions += 1
                    rewards[poi_id] = -10 * n_incursions
                elif poi_rewards[poi_id] > rewards[poi_id] and not rd.pois["P{0}".format(poi_id)].hazardous:
                    rewards[poi_id] = poi_rewards[poi_id]

        reward_history.append(sum(rewards))
        incursion_tracker.append(n_incursions)
        average_reward += sum(rewards)
        srun += 1

    print(average_reward/stat_runs)
    create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths")
    create_csv_file(reward_history, "Output_Data/", "Final_GlobalRewards.csv")
    create_csv_file(incursion_tracker, "Output_Data/", "HazardIncursions.csv")
    if p["vis_running"]:
        run_visualizer()


if __name__ == '__main__':
    test_trained_policy()
