# For Python Code
# import Python_Code.ccea as ccea
# import Python_Code.neural_net as neural_network
# from Python_Code.homogeneous_rewards import calc_global, calc_difference, calc_dpp
# from AADI_RoverDomain.rover_domain_python import RoverDomain

# For Cython Code
import pyximport; pyximport.install(language_level=3)
from ccea import Ccea
from neural_network import NeuralNetwork
from homogeneous_rewards import calc_global, calc_difference, calc_dpp, calc_sdpp
from rover_domain_cython import RoverDomain
from rover import Rover

from AADI_RoverDomain.parameters import Parameters
import csv; import os; import sys
import numpy as np


def save_reward_history(reward_history, file_name):
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)

def save_rover_configuration(rovers, nrovers):
        """
        Saves world configuration to a txt files in a folder called Output_Data
        :Output: Three txt files for Rover starting positions, POI postions, and POI values
        """
        dir_name = 'Output_Data/'  # Intended directory for output files

        if not os.path.exists(dir_name):  # If Data directory does not exist, create it
            os.makedirs(dir_name)

        pfile_name = os.path.join(dir_name, 'Rover_Config.csv')

        row = np.zeros(3)
        with open(pfile_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for rov_id in range(nrovers):
                row[0] = rovers["Rover{0}".format(rov_id)].rover_x
                row[1] = rovers["Rover{0}".format(rov_id)].rover_y
                row[2] = rovers["Rover{0}".format(rov_id)].rover_theta
                writer.writerow(row[:])

def save_rover_path(p, rover_path):  # Save path rovers take using best policy found
    dir_name = 'Output_Data/'  # Intended directory for output files
    nrovers = p.num_rovers

    rpath_name = os.path.join(dir_name, 'Rover_Paths.txt')

    rpath = open(rpath_name, 'a')
    for rov_id in range(nrovers):
        for t in range(p.num_steps+1):
            rpath.write('%f' % rover_path[t, rov_id, 0])
            rpath.write('\t')
            rpath.write('%f' % rover_path[t, rov_id, 1])
            rpath.write('\t')
        rpath.write('\n')
    rpath.write('\n')
    rpath.close()


# HOMOGENEOUS ROVER TEAMS ---------------------------------------------------------------------------------------------
def run_homogeneous_rovers():
    # For Python code
    # p = Parameters()
    # cc = ccea.Ccea(p)
    # nn = neural_network.NeuralNetwork(p)
    # rd = RoverDomain(p)

    # For Cython Code
    p = Parameters()
    cc = Ccea(p)
    nn = NeuralNetwork(p)
    rd = RoverDomain(p)

    # Create instances of rovers
    rovers = {}
    for rover_id in range(p.num_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(p, rover_id)

    if p.new_world_config == 1:
        save_rover_configuration(rovers, p.num_rovers)

    # Checks to make sure gen switch and step switch are not both engaged
    if p.gen_suggestion_switch and p.step_suggestion_switch:
        sys.exit('Gen Switch and Step Switch are both True')

    print("Reward Type: ", p.reward_type)
    print("Coupling Requirement: ", p.coupling)

    for srun in range(p.stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        rd.inital_world_setup()
        cc.reset_populations()  # Randomly initialize ccea populations
        nn.reset_nn()  # Initialize NN architecture
        suggestion = p.suggestion_type
        reward_history = []

        for gen in range(p.generations):
            print("Gen: %i" % gen)
            if p.gen_suggestion_switch and gen == p.gen_switch_point:
                if p.reward_type == "SDPP":
                    suggestion = p.new_suggestion

            cc.select_policy_teams()
            for team_number in range(cc.total_pop_size):  # Each policy in CCEA is tested in teams
                rd.clear_rover_path()
                for rover_id in range(p.num_rovers):
                    rovers["Rover{0}".format(rover_id)].reset_rover()
                    rd.rover_path[0, rover_id, 0] = rovers["Rover{0}".format(rover_id)].rover_x
                    rd.rover_path[0, rover_id, 1] = rovers["Rover{0}".format(rover_id)].rover_y
                    rd.rover_path[0, rover_id, 2] = rovers["Rover{0}".format(rover_id)].rover_theta
                for rover_id in range(p.num_rovers):
                    rovers["Rover{0}".format(rover_id)].get_joint_state(rovers, rd.pois, p.num_rovers, p.num_pois)
                for steps in range(p.num_steps):
                    for rover_id in range(p.num_rovers):
                        policy_id = int(cc.team_selection[rover_id, team_number])
                        nn.run_neural_network(rovers["Rover{0}".format(rover_id)].sensor_readings, cc.pops[rover_id, policy_id], rover_id)
                        rovers["Rover{0}".format(rover_id)].step(nn.out_layer, p.x_dim, p.y_dim)
                        rd.rover_path[steps+1, rover_id, 0] = rovers["Rover{0}".format(rover_id)].rover_x
                        rd.rover_path[steps+1, rover_id, 1] = rovers["Rover{0}".format(rover_id)].rover_y
                        rd.rover_path[steps+1, rover_id, 2] = rovers["Rover{0}".format(rover_id)].rover_theta
                    for rover_id in range(p.num_rovers):
                        rovers["Rover{0}".format(rover_id)].get_joint_state(rovers, rd.pois, p.num_rovers, p.num_pois)

                # Update fitness of policies using reward information
                global_reward = calc_global(p, rd.rover_path, rd.pois)
                if p.reward_type == "Global":
                    for rover_id in range(p.num_rovers):
                        policy_id = int(cc.team_selection[rover_id, team_number])
                        cc.fitness[rover_id, policy_id] = global_reward
                elif p.reward_type == "Difference":
                    d_reward = calc_difference(p, rd.rover_path, rd.pois, global_reward)
                    for rover_id in range(p.num_rovers):
                        policy_id = int(cc.team_selection[rover_id, team_number])
                        cc.fitness[rover_id, policy_id] = d_reward[rover_id]
                elif p.reward_type == "DPP":
                    dpp_reward = calc_dpp(p, rd.rover_path, rd.pois, global_reward)
                    for rover_id in range(p.num_rovers):
                        policy_id = int(cc.team_selection[rover_id, team_number])
                        cc.fitness[rover_id, policy_id] = dpp_reward[rover_id]
                elif p.reward_type == "SDPP":
                    sdpp_reward = calc_sdpp(p, rd.rover_path, rd.pois, global_reward, suggestion)
                    for rover_id in range(p.num_rovers):
                        policy_id = int(cc.team_selection[rover_id, team_number])
                        cc.fitness[rover_id, policy_id] = sdpp_reward[rover_id]
                else:
                    sys.exit('Incorrect Reward Type')

            # Testing Phase (test best policies found so far)
            rd.clear_rover_path()
            for rover_id in range(p.num_rovers):
                rovers["Rover{0}".format(rover_id)].reset_rover()
                rd.rover_path[0, rover_id, 0] = rovers["Rover{0}".format(rover_id)].rover_x
                rd.rover_path[0, rover_id, 1] = rovers["Rover{0}".format(rover_id)].rover_y
                rd.rover_path[0, rover_id, 2] = rovers["Rover{0}".format(rover_id)].rover_theta
            for rover_id in range(p.num_rovers):
                rovers["Rover{0}".format(rover_id)].get_joint_state(rovers, rd.pois, p.num_rovers, p.num_pois)
            for steps in range(p.num_steps):
                for rover_id in range(p.num_rovers):
                    policy_id = np.argmax(cc.fitness[rover_id])
                    nn.run_neural_network(rovers["Rover{0}".format(rover_id)].sensor_readings, cc.pops[rover_id, policy_id], rover_id)
                    rovers["Rover{0}".format(rover_id)].step(nn.out_layer, p.x_dim, p.y_dim)
                    rd.rover_path[steps+1, rover_id, 0] = rovers["Rover{0}".format(rover_id)].rover_x
                    rd.rover_path[steps+1, rover_id, 1] = rovers["Rover{0}".format(rover_id)].rover_y
                    rd.rover_path[steps+1, rover_id, 2] = rovers["Rover{0}".format(rover_id)].rover_theta
                for rover_id in range(p.num_rovers):
                    rovers["Rover{0}".format(rover_id)].get_joint_state(rovers, rd.pois, p.num_rovers, p.num_pois)

            global_reward = calc_global(p, rd.rover_path, rd.pois)
            reward_history.append(global_reward)

            if gen == (p.generations-1):  # Save path at end of final generation
                save_rover_path(p, rd.rover_path)

            cc.down_select()  # Choose new parents and create new offspring population

        if p.reward_type == "Global":
            save_reward_history(reward_history, "Global_Reward.csv")
        if p.reward_type == "Difference":
            save_reward_history(reward_history, "Difference_Reward.csv")
        if p.reward_type == "DPP":
            save_reward_history(reward_history, "DPP_Reward.csv")
        if p.reward_type == "SDPP":
            save_reward_history(reward_history, "SDPP_Reward.csv")


def main():
    run_homogeneous_rovers()

main()  # Run the program
