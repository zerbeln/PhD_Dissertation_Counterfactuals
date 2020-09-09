import pyximport; pyximport.install(language_level=3)
from Python_Code.ccea import Ccea
from Python_Code.gru_network import NeuralNetwork
from Python_Code.reward_functions import calc_global, calc_difference, calc_dpp
from Python_Code.suggestion_rewards import calc_sdpp, calc_sd_reward, sdpp_and_sd
from Python_Code.rover_domain import RoverDomain
from Python_Code.agent import Rover

from parameters import parameters as p
import csv; import os; import sys
import numpy as np


def save_reward_history(reward_history, file_name):
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


def save_rover_configuration(rovers):
    """
    Saves rover positions to a csv file in a folder called Output_Data
    :Output: CSV file containing rover starting positions
    """
    dir_name = 'Output_Data/'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    pfile_name = os.path.join(dir_name, 'Rover_Config.csv')

    row = np.zeros(3)
    with open(pfile_name, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for rov_id in range(p["n_rovers"]):
            row[0] = rovers["Rover{0}".format(rov_id)].rover_x
            row[1] = rovers["Rover{0}".format(rov_id)].rover_y
            row[2] = rovers["Rover{0}".format(rov_id)].rover_theta
            writer.writerow(row[:])


def save_rover_path(rover_path):  # Save path rovers take using best policy found
    dir_name = 'Output_Data/'  # Intended directory for output files

    rpath_name = os.path.join(dir_name, 'Rover_Paths.txt')

    rpath = open(rpath_name, 'a')
    for rov_id in range(p["n_rovers"]):
        for t in range(p["n_steps"]+1):
            rpath.write('%f' % rover_path[t, rov_id, 0])
            rpath.write('\t')
            rpath.write('%f' % rover_path[t, rov_id, 1])
            rpath.write('\t')
        rpath.write('\n')
    rpath.write('\n')
    rpath.close()


def rovers_global_only(reward_type):

    rd = RoverDomain()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["NN{0}".format(rover_id)] = NeuralNetwork()
        rovers["EA{0}".format(rover_id)] = Ccea()

    # Save rover starting positions when a new configuration is created
    if p["new_world_config"] == 1:
        save_rover_configuration(rovers)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["coupling"])

    for srun in range(p["stat_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        rd.inital_world_setup(rovers)
        for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
            rovers["EA{0}".format(rover_id)].create_new_population()
        reward_history = []

        for gen in range(p["generations"]):
            # print("Gen: %i" % gen)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for rover_id in range(p["n_rovers"]):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    rovers["NN{0}".format(rover_id)].reset_nn()
                rd.update_rover_path(rovers, -1)  # Record starting position of each rover
                for steps in range(p["n_steps"]):
                    for rover_id in range(p["n_rovers"]):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                    for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["NN{0}".format(rover_id)].get_weights(rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)])
                        state_vec = rovers["Rover{0}".format(rover_id)].sensor_readings
                        mem_block = rovers["Rover{0}".format(rover_id)].mem_block
                        rovers["NN{0}".format(rover_id)].run_neural_network(state_vec, mem_block)
                        nn_wgate = rovers["NN{0}".format(rover_id)].wgate_outputs
                        nn_enc_mem = rovers["NN{0}".format(rover_id)].encoded_memory
                        rovers["Rover{0}".format(rover_id)].update_memory(nn_wgate, nn_enc_mem)
                        rovers["Rover{0}".format(rover_id)].step(rovers["NN{0}".format(rover_id)].out_layer)
                    rd.update_rover_path(rovers, steps)

                # Update fitness of policies using reward information
                global_reward = calc_global(rd.rover_path, rd.pois)
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = global_reward

            # Testing Phase (test best policies found so far)
            rd.clear_rover_path()
            for rover_id in range(p["n_rovers"]):
                rovers["Rover{0}".format(rover_id)].reset_rover()
            rd.update_rover_path(rovers, -1)
            for steps in range(p["n_steps"]):
                for rover_id in range(p["n_rovers"]):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                for rover_id in range(p["n_rovers"]):  # Rover processes information froms can and acts
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    rovers["NN{0}".format(rover_id)].get_weights(rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)])
                    state_vec = rovers["Rover{0}".format(rover_id)].sensor_readings
                    mem_block = rovers["Rover{0}".format(rover_id)].mem_block
                    rovers["NN{0}".format(rover_id)].run_neural_network(state_vec, mem_block)
                    nn_wgate = rovers["NN{0}".format(rover_id)].wgate_outputs
                    nn_enc_mem = rovers["NN{0}".format(rover_id)].encoded_memory
                    rovers["Rover{0}".format(rover_id)].update_memory(nn_wgate, nn_enc_mem)
                    rovers["Rover{0}".format(rover_id)].step(rovers["NN{0}".format(rover_id)].out_layer)
                rd.update_rover_path(rovers, steps)

            global_reward = calc_global(rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                save_rover_path(rd.rover_path)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "Global_Reward.csv")

def rovers_difference_rewards(reward_type):
    rd = RoverDomain()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["NN{0}".format(rover_id)] = NeuralNetwork()
        rovers["EA{0}".format(rover_id)] = Ccea()

    # Save rover starting positions when a new configuration is created
    if p.new_world_config == 1:
        save_rover_configuration(rovers)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["coupling"])

    for srun in range(p["stat_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        rd.inital_world_setup(rovers)
        for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
            rovers["EA{0}".format(rover_id)].create_new_population()
        reward_history = []

        for gen in range(p["generations"]):
            # print("Gen: %i" % gen)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for rover_id in range(p["n_rovers"]):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    rovers["NN{0}".format(rover_id)].reset_nn()
                rd.update_rover_path(rovers, -1)  # Record starting position of each rover
                for steps in range(p["n_steps"]):
                    for rover_id in range(p["n_rovers"]):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                    for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["NN{0}".format(rover_id)].get_weights(rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)])
                        state_vec = rovers["Rover{0}".format(rover_id)].sensor_readings
                        mem_block = rovers["Rover{0}".format(rover_id)].mem_block
                        rovers["NN{0}".format(rover_id)].run_neural_network(state_vec, mem_block)
                        nn_wgate = rovers["NN{0}".format(rover_id)].wgate_outputs
                        nn_enc_mem = rovers["NN{0}".format(rover_id)].encoded_memory
                        rovers["Rover{0}".format(rover_id)].update_memory(nn_wgate, nn_enc_mem)
                        rovers["Rover{0}".format(rover_id)].step(rovers["NN{0}".format(rover_id)].out_layer)
                    rd.update_rover_path(rovers, steps)

                # Update fitness of policies using reward information
                global_reward = calc_global(rd.rover_path, rd.pois)
                d_reward = calc_difference(rd.rover_path, rd.pois, global_reward)
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = d_reward[rover_id]


            # Testing Phase (test best policies found so far)
            rd.clear_rover_path()
            for rover_id in range(p["n_rovers"]):
                rovers["Rover{0}".format(rover_id)].reset_rover()
            rd.update_rover_path(rovers, -1)
            for steps in range(p["n_steps"]):
                for rover_id in range(p["n_rovers"]):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                for rover_id in range(p["n_rovers"]):  # Rover processes information froms can and acts
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    rovers["NN{0}".format(rover_id)].get_weights(rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)])
                    state_vec = rovers["Rover{0}".format(rover_id)].sensor_readings
                    mem_block = rovers["Rover{0}".format(rover_id)].mem_block
                    rovers["NN{0}".format(rover_id)].run_neural_network(state_vec, mem_block)
                    nn_wgate = rovers["NN{0}".format(rover_id)].wgate_outputs
                    nn_enc_mem = rovers["NN{0}".format(rover_id)].encoded_memory
                    rovers["Rover{0}".format(rover_id)].update_memory(nn_wgate, nn_enc_mem)
                    rovers["Rover{0}".format(rover_id)].step(rovers["NN{0}".format(rover_id)].out_layer)
                rd.update_rover_path(rovers, steps)

            global_reward = calc_global(rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                save_rover_path(rd.rover_path)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "Difference_Reward.csv")

def rovers_suggestions_difference(reward_type, suggestion_type):
    rd = RoverDomain()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["NN{0}".format(rover_id)] = NeuralNetwork()
        rovers["EA{0}".format(rover_id)] = Ccea()

    # Save rover starting positions when a new configuration is created
    if p.new_world_config == 1:
        save_rover_configuration(rovers)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["coupling"])

    for srun in range(p["stat_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        rd.inital_world_setup(rovers)
        for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
            rovers["EA{0}".format(rover_id)].create_new_population()
        reward_history = []

        for gen in range(p["generations"]):
            # print("Gen: %i" % gen)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for rover_id in range(p["n_rovers"]):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    rovers["NN{0}".format(rover_id)].reset_nn()
                rd.update_rover_path(rovers, -1)  # Record starting position of each rover
                for steps in range(p["n_steps"]):
                    for rover_id in range(p["n_rovers"]):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                    for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["NN{0}".format(rover_id)].get_weights(rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)])
                        state_vec = rovers["Rover{0}".format(rover_id)].sensor_readings
                        mem_block = rovers["Rover{0}".format(rover_id)].mem_block
                        rovers["NN{0}".format(rover_id)].run_neural_network(state_vec, mem_block)
                        nn_wgate = rovers["NN{0}".format(rover_id)].wgate_outputs
                        nn_enc_mem = rovers["NN{0}".format(rover_id)].encoded_memory
                        rovers["Rover{0}".format(rover_id)].update_memory(nn_wgate, nn_enc_mem)
                        rovers["Rover{0}".format(rover_id)].step(rovers["NN{0}".format(rover_id)].out_layer)
                    rd.update_rover_path(rovers, steps)

                # Update fitness of policies using reward information
                global_reward = calc_global(rd.rover_path, rd.pois)
                d_reward = calc_sd_reward(rd.rover_path, rd.pois, global_reward, suggestion_type)
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = d_reward[rover_id]

            # Testing Phase (test best policies found so far)
            rd.clear_rover_path()
            for rover_id in range(p["n_rovers"]):
                rovers["Rover{0}".format(rover_id)].reset_rover()
            rd.update_rover_path(rovers, -1)
            for steps in range(p["n_steps"]):
                for rover_id in range(p["n_rovers"]):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                for rover_id in range(p["n_rovers"]):  # Rover processes information froms can and acts
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    rovers["NN{0}".format(rover_id)].get_weights(rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)])
                    state_vec = rovers["Rover{0}".format(rover_id)].sensor_readings
                    mem_block = rovers["Rover{0}".format(rover_id)].mem_block
                    rovers["NN{0}".format(rover_id)].run_neural_network(state_vec, mem_block)
                    nn_wgate = rovers["NN{0}".format(rover_id)].wgate_outputs
                    nn_enc_mem = rovers["NN{0}".format(rover_id)].encoded_memory
                    rovers["Rover{0}".format(rover_id)].update_memory(nn_wgate, nn_enc_mem)
                    rovers["Rover{0}".format(rover_id)].step(rovers["NN{0}".format(rover_id)].out_layer)
                rd.update_rover_path(rovers, steps)

            global_reward = calc_global(rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                save_rover_path(rd.rover_path)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "SDIF_Reward.csv")

def rovers_dplusplus_rewards(reward_type):
    rd = RoverDomain()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["NN{0}".format(rover_id)] = NeuralNetwork()
        rovers["EA{0}".format(rover_id)] = Ccea()

    # Save rover starting positions when a new configuration is created
    if p.new_world_config == 1:
        save_rover_configuration(rovers)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["coupling"])

    for srun in range(p["stat_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        rd.inital_world_setup(rovers)
        for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
            rovers["EA{0}".format(rover_id)].create_new_population()
        reward_history = []

        for gen in range(p["generations"]):
            # print("Gen: %i" % gen)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for rover_id in range(p["n_rovers"]):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    rovers["NN{0}".format(rover_id)].reset_nn()
                rd.update_rover_path(rovers, -1)  # Record starting position of each rover
                for steps in range(p["n_steps"]):
                    for rover_id in range(p["n_rovers"]):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                    for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["NN{0}".format(rover_id)].get_weights(rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)])
                        state_vec = rovers["Rover{0}".format(rover_id)].sensor_readings
                        mem_block = rovers["Rover{0}".format(rover_id)].mem_block
                        rovers["NN{0}".format(rover_id)].run_neural_network(state_vec, mem_block)
                        nn_wgate = rovers["NN{0}".format(rover_id)].wgate_outputs
                        nn_enc_mem = rovers["NN{0}".format(rover_id)].encoded_memory
                        rovers["Rover{0}".format(rover_id)].update_memory(nn_wgate, nn_enc_mem)
                        rovers["Rover{0}".format(rover_id)].step(rovers["NN{0}".format(rover_id)].out_layer)
                    rd.update_rover_path(rovers, steps)

                # Update fitness of policies using reward information
                global_reward = calc_global(rd.rover_path, rd.pois)
                dpp_reward = calc_dpp(rd.rover_path, rd.pois, global_reward)
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = dpp_reward[rover_id]

            # Testing Phase (test best policies found so far)
            rd.clear_rover_path()
            for rover_id in range(p["n_rovers"]):
                rovers["Rover{0}".format(rover_id)].reset_rover()
            rd.update_rover_path(rovers, -1)
            for steps in range(p["n_steps"]):
                for rover_id in range(p["n_rovers"]):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                for rover_id in range(p["n_rovers"]):  # Rover processes information froms can and acts
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    rovers["NN{0}".format(rover_id)].get_weights(rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)])
                    state_vec = rovers["Rover{0}".format(rover_id)].sensor_readings
                    mem_block = rovers["Rover{0}".format(rover_id)].mem_block
                    rovers["NN{0}".format(rover_id)].run_neural_network(state_vec, mem_block)
                    nn_wgate = rovers["NN{0}".format(rover_id)].wgate_outputs
                    nn_enc_mem = rovers["NN{0}".format(rover_id)].encoded_memory
                    rovers["Rover{0}".format(rover_id)].update_memory(nn_wgate, nn_enc_mem)
                    rovers["Rover{0}".format(rover_id)].step(rovers["NN{0}".format(rover_id)].out_layer)
                rd.update_rover_path(rovers, steps)

            global_reward = calc_global(rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                save_rover_path(rd.rover_path)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "DPP_Reward.csv")

def rovers_suggestions_dpp(reward_type, suggestion_type):
    rd = RoverDomain()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["NN{0}".format(rover_id)] = NeuralNetwork()
        rovers["EA{0}".format(rover_id)] = Ccea()

    # Save rover starting positions when a new configuration is created
    if p.new_world_config == 1:
        save_rover_configuration(rovers)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["coupling"])

    for srun in range(p["stat_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        rd.inital_world_setup(rovers)
        for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
            rovers["EA{0}".format(rover_id)].create_new_population()
        suggestion = suggestion_type
        reward_history = []

        for gen in range(p["generations"]):
            # print("Gen: %i" % gen)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for rover_id in range(p["n_rovers"]):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    rovers["NN{0}".format(rover_id)].reset_nn()
                rd.update_rover_path(rovers, -1)  # Record starting position of each rover
                for steps in range(p["n_steps"]):
                    for rover_id in range(p["n_rovers"]):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                    for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["NN{0}".format(rover_id)].get_weights(rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)])
                        state_vec = rovers["Rover{0}".format(rover_id)].sensor_readings
                        mem_block = rovers["Rover{0}".format(rover_id)].mem_block
                        rovers["NN{0}".format(rover_id)].run_neural_network(state_vec, mem_block)
                        nn_wgate = rovers["NN{0}".format(rover_id)].wgate_outputs
                        nn_enc_mem = rovers["NN{0}".format(rover_id)].encoded_memory
                        rovers["Rover{0}".format(rover_id)].update_memory(nn_wgate, nn_enc_mem)
                        rovers["Rover{0}".format(rover_id)].step(rovers["NN{0}".format(rover_id)].out_layer)
                    rd.update_rover_path(rovers, steps)

                # Update fitness of policies using reward information
                global_reward = calc_global(rd.rover_path, rd.pois)
                sdpp_reward = calc_sdpp(rd.rover_path, rd.pois, global_reward, suggestion)
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = sdpp_reward[rover_id]

            # Testing Phase (test best policies found so far)
            rd.clear_rover_path()
            for rover_id in range(p["n_rovers"]):
                rovers["Rover{0}".format(rover_id)].reset_rover()
            rd.update_rover_path(rovers, -1)
            for steps in range(p["n_steps"]):
                for rover_id in range(p["n_rovers"]):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                for rover_id in range(p["n_rovers"]):  # Rover processes information froms can and acts
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    rovers["NN{0}".format(rover_id)].get_weights(rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)])
                    state_vec = rovers["Rover{0}".format(rover_id)].sensor_readings
                    mem_block = rovers["Rover{0}".format(rover_id)].mem_block
                    rovers["NN{0}".format(rover_id)].run_neural_network(state_vec, mem_block)
                    nn_wgate = rovers["NN{0}".format(rover_id)].wgate_outputs
                    nn_enc_mem = rovers["NN{0}".format(rover_id)].encoded_memory
                    rovers["Rover{0}".format(rover_id)].update_memory(nn_wgate, nn_enc_mem)
                    rovers["Rover{0}".format(rover_id)].step(rovers["NN{0}".format(rover_id)].out_layer)
                rd.update_rover_path(rovers, steps)

            global_reward = calc_global(rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                save_rover_path(rd.rover_path)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "SDPP_Reward.csv")

def rover_sdpp_and_sd(reward_type, suggestion_type):
    rd = RoverDomain()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
        rovers["NN{0}".format(rover_id)] = NeuralNetwork()
        rovers["EA{0}".format(rover_id)] = Ccea()

    # Save rover starting positions when a new configuration is created
    if p.new_world_config == 1:
        save_rover_configuration(rovers)

    print("Reward Type: ", reward_type)
    print("Coupling Requirement: ", p["coupling"])

    for srun in range(p["stat_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        rd.inital_world_setup(rovers)
        for rover_id in range(p["n_rovers"]):  # Randomly initialize ccea populations
            rovers["EA{0}".format(rover_id)].create_new_population()
        suggestion = suggestion_type
        reward_history = []

        for gen in range(p["generations"]):
            # print("Gen: %i" % gen)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for rover_id in range(p["n_rovers"]):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    rovers["NN{0}".format(rover_id)].reset_nn()
                rd.update_rover_path(rovers, -1)  # Record starting position of each rover
                for steps in range(p["n_steps"]):
                    for rover_id in range(p["n_rovers"]):  # Rover scans environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                    for rover_id in range(p["n_rovers"]):  # Rover processes scan information and acts
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["NN{0}".format(rover_id)].get_weights(rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)])
                        state_vec = rovers["Rover{0}".format(rover_id)].sensor_readings
                        mem_block = rovers["Rover{0}".format(rover_id)].mem_block
                        rovers["NN{0}".format(rover_id)].run_neural_network(state_vec, mem_block)
                        nn_wgate = rovers["NN{0}".format(rover_id)].wgate_outputs
                        nn_enc_mem = rovers["NN{0}".format(rover_id)].encoded_memory
                        rovers["Rover{0}".format(rover_id)].update_memory(nn_wgate, nn_enc_mem)
                        rovers["Rover{0}".format(rover_id)].step(rovers["NN{0}".format(rover_id)].out_layer)
                    rd.update_rover_path(rovers, steps)

                # Update fitness of policies using reward information
                global_reward = calc_global(rd.rover_path, rd.pois)
                sdpp_reward = sdpp_and_sd(rd.rover_path, rd.pois, global_reward, suggestion)
                for rover_id in range(p["n_rovers"]):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] = sdpp_reward[rover_id]

            # Testing Phase (test best policies found so far)
            rd.clear_rover_path()
            for rover_id in range(p["n_rovers"]):
                rovers["Rover{0}".format(rover_id)].reset_rover()
            rd.update_rover_path(rovers, -1)
            for steps in range(p["n_steps"]):
                for rover_id in range(p["n_rovers"]):  # Rover scans environment
                    rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois)
                for rover_id in range(p["n_rovers"]):  # Rover processes information froms can and acts
                    policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
                    rovers["NN{0}".format(rover_id)].get_weights(rovers["EA{0}".format(rover_id)].population["pop{0}".format(policy_id)])
                    state_vec = rovers["Rover{0}".format(rover_id)].sensor_readings
                    mem_block = rovers["Rover{0}".format(rover_id)].mem_block
                    rovers["NN{0}".format(rover_id)].run_neural_network(state_vec, mem_block)
                    nn_wgate = rovers["NN{0}".format(rover_id)].wgate_outputs
                    nn_enc_mem = rovers["NN{0}".format(rover_id)].encoded_memory
                    rovers["Rover{0}".format(rover_id)].update_memory(nn_wgate, nn_enc_mem)
                    rovers["Rover{0}".format(rover_id)].step(rovers["NN{0}".format(rover_id)].out_layer)
                rd.update_rover_path(rovers, steps)

            global_reward = calc_global(rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                save_rover_path(rd.rover_path)

            for rover_id in range(p["n_rovers"]):
                rovers["EA{0}".format(rover_id)].down_select()  # Choose new parents and create new offspring population

        save_reward_history(reward_history, "SDPP_SD_Reward.csv")


def main(reward_type="Global", suggestion="none"):
    """
    reward_type: Global, Difference, SDIF, DPP, SDPP
    Suggestions: high_val, low_val, high_low, val_based, or none (none is standard D++)_
    :param suggestion:
    :return:
    """

    if reward_type == "Global":
        rovers_global_only(reward_type)
    elif reward_type == "Difference":
        rovers_difference_rewards(reward_type)
    elif reward_type == "SDIF":
        rovers_suggestions_difference(reward_type, suggestion)
    elif reward_type == "DPP":
        rovers_dplusplus_rewards(reward_type)
    elif reward_type == "SDPP":
        rovers_suggestions_dpp(reward_type, suggestion)
    elif reward_type == "SD_AND_SDPP":
        rover_sdpp_and_sd(reward_type, suggestion)
    else:
        sys.exit('Incorrect Reward Type')

main(reward_type="Global", suggestion="none")  # Run the program
