import pyximport; pyximport.install()
from ccea import Ccea
from neural_network import NeuralNetwork
from parameters import Parameters as p
from rover_domain_w_setup import *
from rover_domain import RoverDomain
from reward import calc_global_reward, calc_difference_reward, calc_dpp_reward, calc_sdpp_reward
import csv

def save_reward_history(reward_history, file_name):
    save_file_name = file_name

    with open(save_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Generation"] + list(range(p.generations)))
        for s in range(p.stat_runs):
                writer.writerow(['Performance'] + reward_history[s])


def main():
    cc = Ccea()
    nn = NeuralNetwork()
    rd = RoverDomain()

    rtype = p.reward_type
    reward_history = [[] for i in range(p.stat_runs)]

    for srun in range(p.stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA, NN, and world for new stat run
        cc.reset_populations()  # Randomly initialize ccea populations
        nn.reset_nn()  # Initialize NN architecture
        rd.reset()  # Resets rovers to initial positions

        for gen in range(p.generations):
            print("Gen: %i" % gen)
            cc.select_policy_teams()  # Selects which policies will be grouped into which teams
            for team_number in range(cc.population_size):  # Each policy in CCEA is tested in teams
                rd.reset()
                rd.update_observations()  # Make observations about initial state

                done = False
                step_count = 0
                while done == False:
                    for rover_id in range(p.num_rovers):
                        policy_id = cc.team_selection[rover_id, team_number]
                        nn.run_neural_network(rd.rover_observations[rover_id], cc.pops[rover_id, policy_id], rover_id)
                    rd.move_rovers(nn.out_layer)
                    rd.rover_position_histories[step_count, ...] = rd.rover_positions.copy()
                    rd.update_observations()
                    step_count += 1
                    if step_count > p.num_steps:
                        done = True

                # Update fitness of policies using reward information
                if rtype == 0:
                    reward = calc_global_reward(rd.rover_position_histories, rd.poi_values, rd.poi_positions)
                    for pop_id in range(p.num_rovers):
                        policy_id = cc.team_selection[pop_id, team_number]
                        cc.fitness[pop_id, policy_id] = reward
                if rtype == 1:
                    reward = calc_difference_reward(rd.rover_position_histories, rd.poi_values, rd.poi_positions)
                    for pop_id in range(p.num_rovers):
                        policy_id = cc.team_selection[pop_id, team_number]
                        cc.fitness[pop_id, policy_id] = reward[pop_id]
                if rtype == 2:
                    reward = calc_dpp_reward(rd.rover_position_histories, rd.poi_values, rd.poi_positions)
                    for pop_id in range(p.num_rovers):
                        policy_id = cc.team_selection[pop_id, team_number]
                        cc.fitness[pop_id, policy_id] = reward[pop_id]
                if rtype == 3:
                    reward = calc_sdpp_reward(rd.rover_position_histories, rd.poi_values, rd.poi_positions)
                    for pop_id in range(p.num_rovers):
                        policy_id = cc.team_selection[pop_id, team_number]
                        cc.fitness[pop_id, policy_id] = reward[pop_id]

            cc.down_select()  # Perform down_selection after each policy has been evaluated

            # Testing Phase
            rd.reset()  # Set mode to test and do not fully reset the world
            rd.update_observations()

            done = False
            step_count = 0
            while done == False:
                for rover_id in range(p.num_rovers):
                    nn.run_neural_network(rd.rover_observations[rover_id], cc.pops[rover_id, 0], rover_id)
                rd.move_rovers(nn.out_layer)
                rd.rover_position_histories[step_count, ...] = rd.rover_positions.copy()
                rd.update_observations()
                step_count += 1
                if step_count > p.num_steps:
                    done = True

            reward = calc_global_reward(rd.rover_position_histories, rd.poi_values, rd.poi_positions)
            reward_history[srun].append(reward)

    if rtype == 0:
        save_reward_history(reward_history, "Global_Reward.csv")
    if rtype == 1:
        save_reward_history(reward_history, "Difference_Reward.csv")
    if rtype == 2:
        save_reward_history(reward_history, "DPP_Reward.csv")
    if rtype == 3:
        save_reward_history(reward_history, 'SDPP_Reward.csv')


main()  # Run the program
