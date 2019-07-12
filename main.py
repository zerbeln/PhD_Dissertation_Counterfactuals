import Python_Code.ccea as ccea
import Python_Code.neural_net as neural_network
from AADI_RoverDomain.parameters import Parameters as p
from AADI_RoverDomain.rover_domain import RoverDomain
import Python_Code.homogeneous_rewards as homr
import csv; import os; import sys
from AADI_RoverDomain.visualizer import visualize


def save_reward_history(reward_history, file_name):
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


def save_world_configuration(rover_positions, poi_positions, poi_vals):
    dir_name = 'Output_Data/'  # Intended directory for output files
    nrovers = p.num_rovers

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
    cc = ccea.Ccea()
    nn = neural_network.NeuralNetwork()
    rd = RoverDomain()

    rtype = p.reward_type

    for srun in range(p.stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)
        reward_history = []

        # Reset CCEA, NN, and world for new stat run
        cc.reset_populations()  # Randomly initialize ccea populations
        nn.reset_nn()  # Initialize NN architecture
        rd.reset_world()  # Re-initialize world

        save_world_configuration(rd.rover_initial_pos, rd.poi_pos, rd.poi_values)

        for gen in range(p.generations):
            # print("Gen: %i" % gen)
            cc.select_policy_teams()  # Selects which policies will be grouped into which teams

            for team_number in range(cc.population_size):  # Each policy in CCEA is tested in teams
                rd.reset_to_init()  # Resets rovers to initial configuration
                global_reward = 0.0; global_max = 0.0

                done = False; rd.istep = 0
                joint_state = rd.get_joint_state()
                while not done:
                    for rover_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[rover_id][team_number])
                        nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)
                    joint_state, done, global_reward = rd.step(nn.out_layer)

                    if global_reward > global_max:
                        global_max = global_reward

                # Update fitness of policies using reward information
                if rtype == "Global":
                    for rover_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[rover_id][team_number])
                        cc.fitness[rover_id, policy_id] = global_max
                elif rtype == "Difference":
                    d_reward = homr.calc_difference(rd.rover_path, rd.poi_values, rd.poi_pos, global_max)
                    for rover_id in range(p.num_rovers):
                        policy_id = int(cc.team_selection[rover_id][team_number])
                        cc.fitness[rover_id, policy_id] = d_reward[rover_id]
                elif rtype == "DPP":
                    dpp_reward = homr.calc_dpp(rd.rover_path, rd.poi_values, rd.poi_pos, global_max)
                    for rover_id in range(p.num_rovers):
                        policy_id = int(cc.team_selection[rover_id][team_number])
                        cc.fitness[rover_id, policy_id] = dpp_reward[rover_id]
                else:
                    sys.exit('Incorrect Reward Type for Homogeneous Teams')

            cc.down_select()  # Perform down_selection after each policy has been evaluated

            # Testing Phase
            rd.reset_to_init()  # Reset rovers to initial positions
            global_reward = 0.0; global_max = 0.0
            done = False; rd.istep = 0
            joint_state = rd.get_joint_state()
            while not done:
                for rover_id in range(rd.num_agents):
                    nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, 0], rover_id)
                joint_state, done, global_reward = rd.step(nn.out_layer)

                if global_reward > global_max:
                    global_max = global_reward

            reward_history.append(global_max)

            if gen == (p.generations-1):  # Save path at end of final generation
                save_rover_path(rd.rover_path)
                if p.visualizer_on:
                    visualize(rd, global_max)

        if rtype == "Global":
            save_reward_history(reward_history, "Global_Reward.csv")
        if rtype == "Difference":
            save_reward_history(reward_history, "Difference_Reward.csv")
        if rtype == 'DPP':
            save_reward_history(reward_history, "DPP_Reward.csv")


def main():
    if p.team_types == 'homogeneous':
        run_homogeneous_rovers()
    else:
        print('ERROR')


main()  # Run the program
