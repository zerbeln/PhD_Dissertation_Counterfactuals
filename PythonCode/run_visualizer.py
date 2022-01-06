from Python_Code.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
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
        writer.writerow(['Performance'] + reward_history)


def save_rover_path(rover_path, file_name):  # Save path rovers take using best policy found
    """
    Records the path each rover takes using best policy from CCEA (used by visualizer)
    :param rover_path:  trajectory tracker
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    rpath_name = os.path.join(dir_name, file_name)
    rover_file = open(rpath_name, 'wb')
    pickle.dump(rover_path, rover_file)
    rover_file.close()


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

    reward_history = []  # Keep track of team performance throughout training
    average_reward = 0
    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs

        # Load Trained Suggestion Interpreter Weights
        for rk in rd.rovers:
            rover_id = rd.rovers[rk].self_id
            rov_weights = load_saved_policies('RoverWeights{0}'.format(rover_id), rover_id, srun)
            rd.rovers[rk].get_weights(rov_weights)

        # Reset Rover
        for rk in rd.rovers:
            rover_id = rd.rovers[rk].self_id
            rd.rovers[rk].reset_rover()
            final_rover_path[srun, rover_id, 0, 0] = rd.rovers[rk].x_pos
            final_rover_path[srun, rover_id, 0, 1] = rd.rovers[rk].y_pos
            final_rover_path[srun, rover_id, 0, 2] = rd.rovers[rk].theta_pos

        # Initial rover scan of environment
        for rk in rd.rovers:
            rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

        g_rewards = np.zeros(rover_steps)
        for step_id in range(rover_steps):
            # Rover takes an action in the world
            for rk in rd.rovers:
                rover_id = rd.rovers[rk].self_id
                rd.rovers[rk].step(rd.world_x, rd.world_y)
                final_rover_path[srun, rover_id, step_id + 1, 0] = rd.rovers[rk].x_pos
                final_rover_path[srun, rover_id, step_id + 1, 1] = rd.rovers[rk].y_pos
                final_rover_path[srun, rover_id, step_id + 1, 2] = rd.rovers[rk].theta_pos

            # Rover scans environment and processes suggestions
            for rk in rd.rovers:
                rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
            rd.update_observer_distances()

            # Calculate Global Reward
            g_rewards[step_id] = rd.calc_global()

        reward_history.append(sum(g_rewards))
        average_reward += sum(g_rewards)
        save_rover_path(final_rover_path, "Rover_Paths")

    average_reward /= stat_runs
    print(average_reward)
    save_reward_history(reward_history, "Final_GlobalRewards.csv")
    run_visualizer()


if __name__ == '__main__':
    # test_trained_policy()
    run_visualizer(v_running=True)