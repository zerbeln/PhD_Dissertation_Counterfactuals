import pyximport; pyximport.install(language_level=3)
from Python_Code.ccea import Ccea
from Python_Code.reward_functions import calc_global, calc_difference, calc_dpp
from Python_Code.suggestion_rewards import calc_sdpp, calc_sd_reward, sdpp_and_sd
from Python_Code.rover_domain import RoverDomain
from Python_Code.agent import Rover
from Visualizer.visualizer import run_visualizer
import pickle

from parameters import parameters as p
import csv; import os; import sys
import numpy as np


def save_reward_history(reward_history, file_name):
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


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


def save_best_policies(network_weights, file_name):
    dir_name = 'Output_Data'
    fpath_name = os.path.join(dir_name, file_name)
    weight_file = open(fpath_name, 'wb')
    pickle.dump(network_weights, weight_file)
    weight_file.close()


def load_saved_policies(file_name):
    dir_name = 'Output_Data'
    fpath_name = os.path.join(dir_name, file_name)
    weight_file = open(fpath_name, 'rb')
    weights = pickle.load(weight_file)
    weight_file.close()

    return weights


def rovers_global_only(reward_type):
    rd = RoverDomain()
    rd.inital_world_setup()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(False)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["coupling"])

    # Create new CCEA pop
    for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
        rovers["EA{0}".format(rover_id)].create_new_population()

    print("Training in Progress")
    for gen in range(p["generations"]):
        print("Gen: %i" % gen)

        for rover_id in range(p["n_rovers"]):
            rovers["EA{0}".format(rover_id)].select_policy_teams()
        for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
            rd.clear_rover_path()
            for rover_id in range(p["n_rovers"]):
                rovers["Rover{0}".format(rover_id)].reset_rover()
                policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                rovers["Rover{0}".format(rover_id)].get_weights(weights)
            rd.update_rover_path(rovers, -1)  # Record starting position of each rover
            for steps in range(p["n_steps"]):
                for rover_id in range(p["n_rovers"]):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, 0)
                for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                    rovers["Rover{0}".format(rover_id)].run_neuro_controller()
                    rovers["Rover{0}".format(rover_id)].step()
                rd.update_rover_path(rovers, steps)

            # Update fitness of policies using reward information
            global_reward = calc_global(rd.rover_path, rd.pois)
            for rover_id in range(p["n_rovers"]):
                policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                rovers["EA{0}".format(rover_id)].fitness[policy_id] = global_reward

        for rover_id in range(p["n_rovers"]):
            rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

    for rover_id in range(p["n_rovers"]):
        policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
        weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
        save_best_policies(weights, "RoverWeights{0}".format(rover_id))
    print("Training Complete")


def rovers_difference_rewards(reward_type, suggestion_type):
    rd = RoverDomain()
    rd.inital_world_setup()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(False)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["coupling"])

    # Create new CCEA population
    for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
        rovers["EA{0}".format(rover_id)].create_new_population()

    print("Training in Progress")
    for gen in range(p["generations"]):
        print("Gen: %i" % gen)

        for rover_id in range(p["n_rovers"]):
            rovers["EA{0}".format(rover_id)].select_policy_teams()
        for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
            rd.clear_rover_path()
            for rover_id in range(p["n_rovers"]):
                rovers["Rover{0}".format(rover_id)].reset_rover()
                policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                rovers["Rover{0}".format(rover_id)].get_weights(weights)
            rd.update_rover_path(rovers, -1)  # Record starting position of each rover
            for steps in range(p["n_steps"]):
                for rover_id in range(p["n_rovers"]):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                    rovers["Rover{0}".format(rover_id)].run_neuro_controller()
                    rovers["Rover{0}".format(rover_id)].step()
                rd.update_rover_path(rovers, steps)

            # Update fitness of policies using reward information
            global_reward = calc_global(rd.rover_path, rd.pois)
            d_reward = calc_difference(rd.rover_path, rd.pois, global_reward)
            for rover_id in range(p["n_rovers"]):
                policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                rovers["EA{0}".format(rover_id)].fitness[policy_id] = d_reward[rover_id]

        for rover_id in range(p["n_rovers"]):
            rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

    for rover_id in range(p["n_rovers"]):
        policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
        weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
        save_best_policies(weights, "RoverWeights{0}".format(rover_id))
    print("Training Complete")



def rovers_suggestions_difference(reward_type, suggestion_type):
    rd = RoverDomain()
    rd.inital_world_setup()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(False)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["coupling"])

    # Create new CCEA population
    for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
        rovers["EA{0}".format(rover_id)].create_new_population()

    print("Training in Progress")
    for gen in range(p["generations"]):
        print("Gen: %i" % gen)

        for rover_id in range(p["n_rovers"]):
            rovers["EA{0}".format(rover_id)].select_policy_teams()
        for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
            rd.clear_rover_path()
            for rover_id in range(p["n_rovers"]):
                rovers["Rover{0}".format(rover_id)].reset_rover()
                policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                rovers["Rover{0}".format(rover_id)].get_weights(weights)
            rd.update_rover_path(rovers, -1)  # Record starting position of each rover
            for steps in range(p["n_steps"]):
                for rover_id in range(p["n_rovers"]):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                    rovers["Rover{0}".format(rover_id)].run_neuro_controller()
                    rovers["Rover{0}".format(rover_id)].step()
                rd.update_rover_path(rovers, steps)

            # Update fitness of policies using reward information
            global_reward = calc_global(rd.rover_path, rd.pois)
            d_reward = calc_sd_reward(rd.rover_path, rd.pois, global_reward, suggestion_type)
            for rover_id in range(p["n_rovers"]):
                policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                rovers["EA{0}".format(rover_id)].fitness[policy_id] = d_reward[rover_id]

        for rover_id in range(p["n_rovers"]):
            rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

    for rover_id in range(p["n_rovers"]):
        policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
        weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
        save_best_policies(weights, "RoverWeights{0}".format(rover_id))
    print("Training Complete")


def rovers_dplusplus_rewards(reward_type, suggestion_type):
    rd = RoverDomain()
    rd.inital_world_setup()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(False)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["coupling"])

    # Create new CCEA population
    for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
        rovers["EA{0}".format(rover_id)].create_new_population()

    print("Training in Progress")
    for gen in range(p["generations"]):
        print("Gen: %i" % gen)

        for rover_id in range(p["n_rovers"]):
            rovers["EA{0}".format(rover_id)].select_policy_teams()
        for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
            rd.clear_rover_path()
            for rover_id in range(p["n_rovers"]):
                rovers["Rover{0}".format(rover_id)].reset_rover()
                policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                rovers["Rover{0}".format(rover_id)].get_weights(weights)
            rd.update_rover_path(rovers, -1)  # Record starting position of each rover
            for steps in range(p["n_steps"]):
                for rover_id in range(p["n_rovers"]):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                    rovers["Rover{0}".format(rover_id)].run_neuro_controller()
                    rovers["Rover{0}".format(rover_id)].step()
                rd.update_rover_path(rovers, steps)

            # Update fitness of policies using reward information
            global_reward = calc_global(rd.rover_path, rd.pois)
            dpp_reward = calc_dpp(rd.rover_path, rd.pois, global_reward)
            for rover_id in range(p["n_rovers"]):
                policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                rovers["EA{0}".format(rover_id)].fitness[policy_id] = dpp_reward[rover_id]

        for rover_id in range(p["n_rovers"]):
            rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

    for rover_id in range(p["n_rovers"]):
        policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
        weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
        save_best_policies(weights, "RoverWeights{0}".format(rover_id))
    print("Training Complete")


def rovers_suggestions_dpp(reward_type):
    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["coupling"])

    rd = RoverDomain()
    rd.inital_world_setup()

    for policy_id in range(p["n_policies"]):
        print("Training Policy: ", policy_id)
        if policy_id == 0:
            suggestion_type = "low_val"
        else:
            suggestion_type = "high_val"
        suggestion = suggestion_type

        # Create dictionary for each instance of rover and corresponding NN and EA population
        rovers = {}
        for rover_id in range(p["n_rovers"]):
            rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
            rovers["Rover{0}".format(rover_id)].initialize_rover()
            rovers["EA{0}".format(rover_id)] = Ccea(False)

        # Create new CCEA population
        for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
            rovers["EA{0}".format(rover_id)].create_new_population()

        for gen in range(p["generations"]):
            # print("Gen: %i" % gen)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for rover_id in range(p["n_rovers"]):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                    rovers["Rover{0}".format(rover_id)].get_weights(weights)
                rd.update_rover_path(rovers, -1)  # Record starting position of each rover
                for steps in range(p["n_steps"]):
                    for rover_id in range(p["n_rovers"]):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                    for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                        rovers["Rover{0}".format(rover_id)].run_neuro_controller()
                        rovers["Rover{0}".format(rover_id)].step()
                    rd.update_rover_path(rovers, steps)

                # Update fitness of policies using reward information
                global_reward = calc_global(rd.rover_path, rd.pois)
                sdpp_reward = calc_sdpp(rd.rover_path, rd.pois, global_reward, suggestion)
                for rover_id in range(p["n_rovers"]):
                    fit_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[fit_id] = sdpp_reward[rover_id]

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        for rover_id in range(p["n_rovers"]):
            best_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(best_id)]
            save_best_policies(weights, "Rover{0}Policy{1}".format(rover_id, policy_id))
    print("Training Complete")


def rover_sdpp_and_sd(reward_type, suggestion_type):
    rd = RoverDomain()
    rd.inital_world_setup()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(False)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["coupling"])

    # Create new CCEA population
    for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
        rovers["EA{0}".format(rover_id)].create_new_population()
    suggestion = suggestion_type

    print("Training In Progress")
    for gen in range(p["generations"]):
        print("Gen: %i" % gen)

        for rover_id in range(p["n_rovers"]):
            rovers["EA{0}".format(rover_id)].select_policy_teams()
        for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
            rd.clear_rover_path()
            for rover_id in range(p["n_rovers"]):
                rovers["Rover{0}".format(rover_id)].reset_rover()
                policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                rovers["Rover{0}".format(rover_id)].get_weights(weights)
            rd.update_rover_path(rovers, -1)  # Record starting position of each rover
            for steps in range(p["n_steps"]):
                for rover_id in range(p["n_rovers"]):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                    rovers["Rover{0}".format(rover_id)].run_neuro_controller()
                    rovers["Rover{0}".format(rover_id)].step()
                rd.update_rover_path(rovers, steps)

            # Update fitness of policies using reward information
            global_reward = calc_global(rd.rover_path, rd.pois)
            sdpp_reward = sdpp_and_sd(rd.rover_path, rd.pois, global_reward, suggestion)
            for rover_id in range(p["n_rovers"]):
                fit_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                rovers["EA{0}".format(rover_id)].fitness[fit_id] = sdpp_reward[rover_id]

        for rover_id in range(p["n_rovers"]):
            rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

    for rover_id in range(p["n_rovers"]):
        policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
        weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
        save_best_policies(weights, "RoverWeights{0}".format(rover_id))

    print("Training Complete")


def visualize_policies(policy_id):
    """
    Visualize the policies that are pre-trained and stored
    """
    rd = RoverDomain()
    rd.inital_world_setup()

    rovers = {}
    for rover_id in range(p["n_rovers"]):
        weights = load_saved_policies("Rover{0}Policy{1}".format(rover_id, policy_id))
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["Rover{0}".format(rover_id)].get_weights(weights)

    rd.update_final_rover_path(0, rovers, -1)
    for steps in range(p["n_steps"]):
        for rover_id in range(p["n_rovers"]):  # Rover scans environment
            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
        for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
            rovers["Rover{0}".format(rover_id)].run_neuro_controller()
            rovers["Rover{0}".format(rover_id)].step()
        rd.update_final_rover_path(0, rovers, steps)

    save_rover_path(rd.final_rover_path)
    run_visualizer()


def visualize_suggestion(pol_id, confidence):
    """
    Visualize the policies that are pre-trained and stored
    """
    rd = RoverDomain()
    rd.inital_world_setup()

    rovers = {}
    suggestion = np.zeros(p["n_policies"])
    suggestion[pol_id] = 1.0
    for rover_id in range(p["n_rovers"]):
        for policy_id in range(p["n_policies"]):
            weights = load_saved_policies("Rover{0}Policy{1}".format(rover_id, policy_id))
            rovers["Rover{0}Policy{1}".format(rover_id, policy_id)] = weights.copy()
        weights = load_saved_policies("SPolicyRover{0}".format(rover_id))
        rovers["SPolicyRover{0}".format(rover_id)] = weights.copy()
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["Rover{0}".format(rover_id)].initialize_rover()

    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)].reset_rover()
        s_weights = rovers["SPolicyRover{0}".format(rover_id)]
        rovers["Rover{0}".format(rover_id)].get_s_weights(s_weights)
    rd.update_final_rover_path(0, rovers, -1)
    for steps in range(p["n_steps"]):
        for rover_id in range(p["n_rovers"]):  # Rover scans environment
            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
        for rover_id in range(p["n_rovers"]):  # Rover processes information froms can and acts
            rovers["Rover{0}".format(rover_id)].run_s_neuro_controller(suggestion, confidence)
            selected_pol = rovers["Rover{0}".format(rover_id)].select_policy()
            weights = rovers["Rover{0}Policy{1}".format(rover_id, selected_pol)]
            rovers["Rover{0}".format(rover_id)].get_weights(weights)
            rovers["Rover{0}".format(rover_id)].run_neuro_controller()
            rovers["Rover{0}".format(rover_id)].step()
        rd.update_final_rover_path(0, rovers, steps)

    save_rover_path(rd.final_rover_path)
    run_visualizer()


def train_suggestion_newtwork():
    """
    Train the network that will be used to interpret suggestions
    """
    rd = RoverDomain()
    rd.inital_world_setup()

    rovers = {}
    for rover_id in range(p["n_rovers"]):
        for policy_id in range(p["n_policies"]):
            weights = load_saved_policies("Rover{0}Policy{1}".format(rover_id, policy_id))
            rovers["Rover{0}Policy{1}".format(rover_id, policy_id)] = weights.copy()
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["Rover{0}".format(rover_id)].initialize_rover()
        rovers["EA{0}".format(rover_id)] = Ccea(True)
        rovers["EA{0}".format(rover_id)].create_new_population()

    poi_target = "NOPE"
    confidence = 1.0

    for sruns in range(p["stat_runs"]):
        print("Stat Run: ", sruns)
        reward_history = []
        for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
            rovers["EA{0}".format(rover_id)].create_new_population()

        for gen in range(p["generations"]):
            # print("Gen: %i" % gen)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].clear_fitness()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rd.clear_poi_visit_list()
                for pol_id in range(p["n_policies"]):
                    suggestion = np.zeros(p["n_policies"])
                    suggestion[pol_id] = 1.0
                    if pol_id == 0:
                        poi_target = "low_val"
                    else:
                        poi_target = "high_val"
                    rd.clear_rover_path()
                    for rover_id in range(p["n_rovers"]):
                        rovers["Rover{0}".format(rover_id)].reset_rover()
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        s_weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
                        rovers["Rover{0}".format(rover_id)].get_s_weights(s_weights)
                    rd.update_rover_path(rovers, -1)  # Record starting position of each rover
                    for steps in range(p["n_steps"]):
                        for rover_id in range(p["n_rovers"]):  # Rover scans environment
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                        for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                            rovers["Rover{0}".format(rover_id)].run_s_neuro_controller(suggestion, confidence)
                            selected_pol = rovers["Rover{0}".format(rover_id)].select_policy()
                            weights = rovers["Rover{0}Policy{1}".format(rover_id, selected_pol)]
                            rovers["Rover{0}".format(rover_id)].get_weights(weights)
                            rovers["Rover{0}".format(rover_id)].run_neuro_controller()
                            rovers["Rover{0}".format(rover_id)].step()
                        rd.update_rover_path(rovers, steps)
                        rd.determine_poi_visits(rovers)

                    # Update fitness of policies using reward information
                    global_reward = calc_global(rd.rover_path, rd.pois)
                    sdpp_reward = calc_sdpp(rd.rover_path, rd.pois, global_reward, poi_target)
                    for rover_id in range(p["n_rovers"]):
                        fit_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[fit_id] += sdpp_reward[rover_id]

                for poi_id in range(p["n_poi"]):
                    for rover_id in range(p["n_rovers"]):
                        if rd.poi_visits[poi_id, rover_id] == 0:
                            fit_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                            rovers["EA{0}".format(rover_id)].fitness[fit_id] = 0.0

                # Testing Phase (test best policies found so far)
                global_reward = 0
                for pol_id in range(p["n_policies"]):
                    suggestion = np.zeros(p["n_policies"])
                    suggestion[pol_id] = 1.0
                    for rover_id in range(p["n_rovers"]):
                        rovers["Rover{0}".format(rover_id)].reset_rover()
                        best_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                        s_weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(best_id)]
                        rovers["Rover{0}".format(rover_id)].get_s_weights(s_weights)
                    rd.update_final_rover_path(0, rovers, -1)
                    for steps in range(p["n_steps"]):
                        for rover_id in range(p["n_rovers"]):  # Rover scans environment
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                        for rover_id in range(p["n_rovers"]):  # Rover processes information froms can and acts
                            rovers["Rover{0}".format(rover_id)].run_s_neuro_controller(suggestion, confidence)
                            selected_pol = rovers["Rover{0}".format(rover_id)].select_policy()
                            weights = rovers["Rover{0}Policy{1}".format(rover_id, selected_pol)]
                            rovers["Rover{0}".format(rover_id)].get_weights(weights)
                            rovers["Rover{0}".format(rover_id)].run_neuro_controller()
                            rovers["Rover{0}".format(rover_id)].step()
                        rd.update_final_rover_path(sruns, rovers, steps)

                    global_reward += calc_global(rd.rover_path, rd.pois)
                reward_history.append(global_reward)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "SuggestionRewards.csv")
        for rover_id in range(p["n_rovers"]):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)]
            save_best_policies(weights, "SPolicyRover{0}".format(rover_id))

    save_rover_path(rd.final_rover_path)
    run_visualizer()


def main(reward_type="Global", suggestion="none"):
    """
    reward_type: Global, Difference, SDIF, DPP, SDPP
    Suggestions: high_val, low_val, high_low, val_based, or none (none is standard D++)_
    :param suggestion:
    :return:
    """

    mode = "train_suggest"  # train, visualize_pol, visualize_sugg, or train _suggest

    if mode == "train":
        if reward_type == "Global":
            rovers_global_only(reward_type)
        elif reward_type == "Difference":
            rovers_difference_rewards(reward_type, suggestion)
        elif reward_type == "SDIF":
            rovers_suggestions_difference(reward_type, suggestion)
        elif reward_type == "DPP":
            rovers_dplusplus_rewards(reward_type, suggestion)
        elif reward_type == "SDPP":
            rovers_suggestions_dpp(reward_type)
        elif reward_type == "SD_AND_SDPP":
            rover_sdpp_and_sd(reward_type, suggestion)
        elif reward_type == "vis":
            run_visualizer()
        else:
            sys.exit('Incorrect Reward Type')
    elif mode == "visualize_pol":
        visualize_policies(1)
    elif mode == "visualize_sugg":
        pol_id = 0
        visualize_suggestion(pol_id, 1)
    else:
        train_suggestion_newtwork()


main(reward_type="SDPP")  # Run the program
