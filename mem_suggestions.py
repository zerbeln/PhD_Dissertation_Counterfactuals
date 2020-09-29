from Python_Code.ccea import GruCcea, Ccea
from Python_Code.reward_functions import calc_global, calc_dpp
from Python_Code.suggestion_rewards import calc_sdpp, calc_sd_reward, sdpp_and_sd
from Python_Code.rover_domain import RoverDomain
from Python_Code.agent import Rover
from Visualizer.visualizer import run_visualizer
import pickle

from parameters import parameters as p
import csv
import os
import numpy as np


def save_reward_history(reward_history, file_name):
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


def save_best_policies(network_weights, file_name):
    dir_name = 'Output_Data'
    fpath_name = os.path.join(dir_name, file_name)
    weight_file = open(fpath_name, 'wb')
    pickle.dump(network_weights, weight_file)
    weight_file.close()


def save_rover_path(rover_path):  # Save path rovers take using best policy found
    """
    Records the path each rover takes using best policy from CCEA (used by visualizer)
    :param rover_path:  trajectory tracker
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files

    rpath_name = os.path.join(dir_name, 'Rover_Paths')
    rover_file = open(rpath_name, 'wb')
    pickle.dump(rover_path, rover_file)
    rover_file.close()


def load_saved_policies(file_name):
    dir_name = 'Output_Data'
    fpath_name = os.path.join(dir_name, file_name)
    weight_file = open(fpath_name, 'rb')
    weights = pickle.load(weight_file)
    weight_file.close()

    return weights


def use_saved_policies(suggestion_type):
    rd = RoverDomain()
    rd.inital_world_setup()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["Rover{0}".format(rover_id)].initialize_rover()

    for srun in range(p["stat_runs"]):
        # Testing Phase (test best policies found so far)
        for rover_id in range(p["n_rovers"]):
            rovers["Rover{0}".format(rover_id)].reset_rover()
            weights = load_saved_policies("RoverWeights{0}".format(rover_id))
            rovers["Rover{0}".format(rover_id)].get_weights(weights)
        rd.update_final_rover_path(srun, rovers, -1)
        for steps in range(p["n_steps"]):
            for rover_id in range(p["n_rovers"]):  # Rover scans environment
                rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, suggestion_type)
            for rover_id in range(p["n_rovers"]):  # Rover processes information froms can and acts
                rovers["Rover{0}".format(rover_id)].run_neuro_controller()
                rovers["Rover{0}".format(rover_id)].step()
            rd.update_final_rover_path(srun, rovers, steps)

        global_reward = calc_global(rd.rover_path, rd.pois)
        print(global_reward)

    save_rover_path(rd.final_rover_path)
    run_visualizer()


def memory_suggestions_dpp(suggestion_type):
    rd = RoverDomain()
    rd.inital_world_setup()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        if p["ctrl_type"] == "GRU":
            rovers["EA{0}".format(rover_id)] = GruCcea()
        elif p["ctrl_type"] == "NN":
            rovers["EA{0}".format(rover_id)] = Ccea()

    print("Coupling Requirement: ", p["coupling"])

    for srun in range(p["stat_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Create new CCEA population
        for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
            rovers["EA{0}".format(rover_id)].create_new_population()
        reward_history = []

        for gen in range(p["generations"]):
            print("Gen: %i" % gen)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for nt in range(p["n_suggestions"]):
                    if nt == 0:
                        suggestion = "low_val"
                    else:
                        suggestion = "high_val"
                    for rover_id in range(p["n_rovers"]):
                        rovers["Rover{0}".format(rover_id)].reset_rover()
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                        rovers["Rover{0}".format(rover_id)].get_weights(weights)
                    rd.update_rover_path(rovers, -1)  # Record starting position of each rover
                    for steps in range(p["n_steps"]):
                        for rover_id in range(p["n_rovers"]):  # Rover scans environment
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, suggestion)
                        for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                            rovers["Rover{0}".format(rover_id)].run_neuro_controller()
                            rovers["Rover{0}".format(rover_id)].step()
                        rd.update_rover_path(rovers, steps)

                    # Update fitness of policies using reward information
                    global_reward = calc_global(rd.rover_path, rd.pois)
                    sdpp_reward = calc_sdpp(rd.rover_path, rd.pois, global_reward, suggestion)
                    for rover_id in range(p["n_rovers"]):
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[policy_id] += sdpp_reward[rover_id]

                for rover_id in range(p["n_rovers"]):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] /= p["n_suggestions"]

            # Testing Phase (test best policies found so far)
            if gen % 5 == 0 or gen == p["generations"] - 1:
                for rover_id in range(p["n_rovers"]):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                    if gen == p["generations"]-1:
                        save_best_policies(weights, "RoverWeights{0}".format(rover_id))
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                rd.update_final_rover_path(srun, rovers, -1)
                for steps in range(p["n_steps"]):
                    for rover_id in range(p["n_rovers"]):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, suggestion_type)
                    for rover_id in range(p["n_rovers"]):  # Rover processes information froms can and acts
                        rovers["Rover{0}".format(rover_id)].run_neuro_controller()
                        rovers["Rover{0}".format(rover_id)].step()
                    rd.update_final_rover_path(srun, rovers, steps)

                global_reward = calc_global(rd.rover_path, rd.pois)
                reward_history.append(global_reward)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "SDPP_Reward.csv")

    save_rover_path(rd.final_rover_path)
    run_visualizer()


def run_rover_domain(train_new_policies=0, suggestion_type="high_val"):
    if train_new_policies == 1:
        memory_suggestions_dpp(suggestion_type)
    elif train_new_policies == 0:
        use_saved_policies(suggestion_type)
    else:
        print("Incorrect Inputs")


run_rover_domain(train_new_policies=1, suggestion_type="low_val")
