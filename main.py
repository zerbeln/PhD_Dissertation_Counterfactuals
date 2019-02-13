import pyximport; pyximport.install()
from ccea import Ccea
from neural_network import NeuralNetwork
from parameters import Parameters as p
from rover_domain_python import RoverDomain
from reward import calc_global_reward, calc_difference_reward, calc_dpp_reward, calc_sdpp_reward
from homogeneous_rewards import calc_global, calc_difference, calc_dpp
import csv; import os; import sys


def save_reward_history(reward_history, file_name):
    dir_name = 'Output_Data/'  # Inteded directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


def save_world_configuration(rover_positions, poi_positions, poi_vals):
    dir_name = 'Output_Data/'  # Inteded directory for output files
    nrovers = p.num_rovers * p.num_types

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    rcoords_name = os.path.join(dir_name, 'Rover_Positions.txt')
    pcoords_name = os.path.join(dir_name, 'POI_Positions.txt')
    pvals_name = os.path.join(dir_name, 'POI_Values.txt')

    rov_coords = open(rcoords_name, 'a')
    for r_id in range(nrovers):  # Record initial rover positions to txt file
        rov_coords.write('%f' % rover_positions[r_id, 0])
        rov_coords.write('\t')
        rov_coords.write('%f' % rover_positions[r_id, 1])
        rov_coords.write('\t')
    rov_coords.write('\n')
    rov_coords.close()

    poi_coords = open(pcoords_name, 'a')
    poi_values = open(pvals_name, 'a')
    for p_id in range(p.num_pois):  # Record POI positions and values
        poi_coords.write('%f' % poi_positions[p_id, 0])
        poi_coords.write('\t')
        poi_coords.write('%f' % poi_positions[p_id, 1])
        poi_coords.write('\t')
        poi_values.write('%f' % poi_vals[p_id])
        poi_values.write('\t')
    poi_coords.write('\n')
    poi_values.write('\n')
    poi_coords.close()
    poi_values.close()


# HETEROGENEOUS ROVER TEAMS -------------------------------------------------------------------------------------------
def run_heterogeneous_rovers():
    cc = Ccea()
    nn = NeuralNetwork()
    rd = RoverDomain()

    rtype = p.reward_type

    for srun in range(p.stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)
        reward_history = []

        # Reset CCEA, NN, and world for new stat run
        cc.reset_populations()  # Randomly initialize ccea populations
        nn.reset_nn()  # Initialize NN architecture
        rd.reset()  # Re-initialize world

        save_world_configuration(rd.rover_initial_pos, rd.poi_pos, rd.poi_value)

        for gen in range(p.generations):
            #  print("Gen: %i" % gen)
            cc.select_policy_teams()  # Selects which policies will be grouped into which teams
            for team_number in range(cc.population_size):  # Each policy in CCEA is tested in teams
                rd.reset_to_init()  # Resets rovers to initial configuration

                done = False
                rd.istep = 0
                joint_state = rd.get_joint_state()
                while done == False:
                    for rover_id in range(rd.num_agents):
                        policy_id = cc.team_selection[rover_id, team_number]
                        nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)
                    joint_state, done = rd.step(nn.out_layer)

                # Update fitness of policies using reward information
                if rtype == 0:
                    reward = calc_global_reward(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = cc.team_selection[pop_id, team_number]
                        cc.fitness[pop_id, policy_id] = reward
                elif rtype == 1:
                    reward = calc_difference_reward(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = cc.team_selection[pop_id, team_number]
                        cc.fitness[pop_id, policy_id] = reward[pop_id]
                elif rtype == 2:
                    reward = calc_dpp_reward(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = cc.team_selection[pop_id, team_number]
                        cc.fitness[pop_id, policy_id] = reward[pop_id]
                elif rtype == 3:
                    reward = calc_sdpp_reward(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = cc.team_selection[pop_id, team_number]
                        cc.fitness[pop_id, policy_id] = reward[pop_id]
                else:
                    sys.exit('Incorrect Reward Type for Heterogeneous Teams')

            cc.down_select()  # Perform down_selection after each policy has been evaluated

            # Testing Phase
            rd.reset_to_init()  # Reset rovers to initial positions

            done = False
            rd.istep = 0
            joint_state = rd.get_joint_state()
            while done == False:
                for rover_id in range(rd.num_agents):
                    nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, 0], rover_id)
                joint_state, done = rd.step(nn.out_layer)

            reward = calc_global_reward(rd.rover_path, rd.poi_value, rd.poi_pos)
            reward_history.append(reward)

        if rtype == 0:
            save_reward_history(reward_history, "Global_Reward.csv")
        if rtype == 1:
            save_reward_history(reward_history, "Difference_Reward.csv")
        if rtype == 2:
            save_reward_history(reward_history, "DPP_Reward.csv")
        if rtype == 3:
            save_reward_history(reward_history, 'SDPP_Reward.csv')


# HOMOGENEOUS ROVER TEAMS ---------------------------------------------------------------------------------------------
def run_homogeneous_rovers():
    cc = Ccea()
    nn = NeuralNetwork()
    rd = RoverDomain()

    rtype = p.reward_type

    for srun in range(p.stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)
        reward_history = []

        # Reset CCEA, NN, and world for new stat run
        cc.reset_populations()  # Randomly initialize ccea populations
        nn.reset_nn()  # Initialize NN architecture
        rd.reset()  # Re-initialize world

        save_world_configuration(rd.rover_initial_pos, rd.poi_pos, rd.poi_value)

        for gen in range(p.generations):
            print("Gen: %i" % gen)
            cc.select_policy_teams()  # Selects which policies will be grouped into which teams
            for team_number in range(cc.population_size):  # Each policy in CCEA is tested in teams
                rd.reset_to_init()  # Resets rovers to initial configuration

                done = False
                rd.istep = 0
                joint_state = rd.get_joint_state()
                while done == False:
                    for rover_id in range(rd.num_agents):
                        policy_id = cc.team_selection[rover_id, team_number]
                        nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)
                    joint_state, done = rd.step(nn.out_layer)

                # Update fitness of policies using reward information
                if rtype == 0:
                    reward = calc_global(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = cc.team_selection[pop_id, team_number]
                        cc.fitness[pop_id, policy_id] = reward
                elif rtype == 1:
                    reward = calc_difference(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = cc.team_selection[pop_id, team_number]
                        cc.fitness[pop_id, policy_id] = reward[pop_id]
                elif rtype == 2:
                    reward = calc_dpp(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = cc.team_selection[pop_id, team_number]
                        cc.fitness[pop_id, policy_id] = reward[pop_id]
                else:
                    sys.exit('Incorrect Reward Type for Homogeneous Teams')

            cc.down_select()  # Perform down_selection after each policy has been evaluated

            # Testing Phase
            rd.reset_to_init()  # Reset rovers to initial positions

            done = False
            rd.istep = 0
            joint_state = rd.get_joint_state()
            while done == False:
                for rover_id in range(rd.num_agents):
                    nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, 0], rover_id)
                joint_state, done = rd.step(nn.out_layer)

            reward = calc_global(rd.rover_path, rd.poi_value, rd.poi_pos)
            reward_history.append(reward)

        if rtype == 0:
            save_reward_history(reward_history, "Global_Reward.csv")
        if rtype == 1:
            save_reward_history(reward_history, "Difference_Reward.csv")
        if rtype == 2:
            save_reward_history(reward_history, "DPP_Reward.csv")


def main():
    if p.rover_types == 'homogeneous':
        run_homogeneous_rovers()
    elif p.rover_types == 'heterogeneous':
        run_heterogeneous_rovers()
    else:
        print('ERROR')


main()  # Run the program
