# For Python Code
# import Python_Code.ccea as ccea
# import Python_Code.neural_net as neural_network
# from Python_Code.homogeneous_rewards import calc_global, calc_difference, calc_dpp

# For Cython Code
import pyximport; pyximport.install()
from Cython_Code.ccea import Ccea
from Cython_Code.neural_network import NeuralNetwork
from Cython_Code.homogeneous_rewards import calc_global, calc_difference, calc_dpp

from AADI_RoverDomain.parameters import Parameters as p
from AADI_RoverDomain.rover_domain import RoverDomain
import csv; import os; import sys
import numpy as np


def save_reward_history(reward_history, file_name):
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)

def save_rover_path(rover_path):  # Save path rovers take using best policy found
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
    # cc = ccea.Ccea()
    # nn = neural_network.NeuralNetwork()
    # rd = RoverDomain()

    # For Cython Code
    cc = Ccea()
    nn = NeuralNetwork()
    rd = RoverDomain()

    rtype = p.reward_type
    rd.inital_world_setup()
    print("Reward Type: ", p.reward_type)
    if p.reward_type == "DPP":
        assert(p.suggestion_type == "none")
        assert(p.gen_suggestion_switch == False)

    for srun in range(p.stat_runs):  # Perform statistical runs
        p.suggestion_type = p.original_suggestion
        print("Run: %i" % srun)
        reward_history = []

        # Reset CCEA and NN new stat run
        cc.reset_populations()  # Randomly initialize ccea populations
        nn.reset_nn()  # Initialize NN architecture

        for gen in range(p.generations):
            # print("Gen: %i" % gen)
            if p.gen_suggestion_switch == True and gen > 499:
                p.suggestion_type = p.new_suggestion  # Switch the suggestion to this
            cc.select_policy_teams()

            for team_number in range(cc.total_pop_size):  # Each policy in CCEA is tested in teams
                rd.reset_to_init()  # Resets rovers to initial configuration
                done = False; rd.istep = 0
                joint_state = rd.get_joint_state()

                while not done:
                    for rover_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[rover_id, team_number])
                        nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)
                    joint_state, done = rd.step(nn.out_layer)

                # Update fitness of policies using reward information
                global_reward = calc_global(rd.rover_path, rd.poi_values, rd.poi_pos)
                if rtype == "Global":
                    for rover_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[rover_id, team_number])
                        cc.fitness[rover_id, policy_id] = global_reward
                elif rtype == "Difference":
                    d_reward = calc_difference(rd.rover_path, rd.poi_values, rd.poi_pos, global_reward)
                    for rover_id in range(p.num_rovers):
                        policy_id = int(cc.team_selection[rover_id, team_number])
                        cc.fitness[rover_id, policy_id] = d_reward[rover_id]
                elif rtype == "DPP" or rtype == "SDPP":
                    dpp_reward = calc_dpp(rd.rover_path, rd.poi_values, rd.poi_pos, global_reward)
                    for rover_id in range(p.num_rovers):
                        policy_id = int(cc.team_selection[rover_id, team_number])
                        cc.fitness[rover_id, policy_id] = dpp_reward[rover_id]
                else:
                    sys.exit('Incorrect Reward Type')

            # Testing Phase (test best policies found so far)
            rd.reset_to_init()  # Reset rovers to initial positions
            done = False; rd.istep = 0
            joint_state = rd.get_joint_state()
            while not done:
                for rover_id in range(rd.num_agents):
                    pol_index = np.argmax(cc.fitness[rover_id])
                    nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, pol_index], rover_id)
                joint_state, done = rd.step(nn.out_layer)

            global_reward = calc_global(rd.rover_path, rd.poi_values, rd.poi_pos)
            reward_history.append(global_reward)

            if gen == (p.generations-1):  # Save path at end of final generation
                save_rover_path(rd.rover_path)

            cc.down_select()  # Choose new parents and create new offspring population

        if rtype == "Global":
            save_reward_history(reward_history, "Global_Reward.csv")
        if rtype == "Difference":
            save_reward_history(reward_history, "Difference_Reward.csv")
        if rtype == "DPP":
            save_reward_history(reward_history, "DPP_Reward.csv")
        if rtype == "SDPP":
            save_reward_history(reward_history, "SDPP_Reward.csv")


def main():
    if p.team_types == 'homogeneous':
        run_homogeneous_rovers()
    else:
        print('ERROR')


main()  # Run the program
