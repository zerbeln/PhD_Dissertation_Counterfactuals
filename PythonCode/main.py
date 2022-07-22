from ccea import Ccea
from cba_network import CBANetwork
from RoverDomain_Core.rover_domain import RoverDomain
from RewardFunctions.cba_rewards import *
from global_functions import *
import random
import numpy as np
from parameters import parameters as p
from custom_rover_skills import get_custom_action
from cba import create_policy_bank, get_counterfactual_state


def train_cba_learned_skills():
    """
    Train suggestions using a pre-trained playbook of rover policies
    """
    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]

    # Suggestion Parameters
    n_suggestions = p["n_suggestions"]
    pbank_type = p["skill_type"]
    s_inp = p["s_inputs"]
    s_hid = p["s_hidden"]
    s_out = p["s_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=s_inp, n_out=s_out, n_hid=s_hid)
        pops["CBA{0}".format(rover_id)] = CBANetwork()

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < stat_runs:
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop and Load Pre-Trained Policies
        for rover_id in range(n_rovers):
            pops["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
            rd.rovers["R{0}".format(rover_id)].policy_bank = create_policy_bank(pbank_type, rover_id, srun)

        policy_rewards = [[] for i in range(n_rovers)]
        for gen in range(generations):
            # Create list of suggestions for rovers to use during training and reset rovers to initial positions
            rover_skills = []
            for rover_id in range(n_rovers):
                pops["EA{0}".format(rover_id)].select_policy_teams()
                pops["EA{0}".format(rover_id)].reset_fitness()
                skill_sample = random.sample(range(n_suggestions), n_suggestions)
                rover_skills.append(skill_sample)

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(population_size):
                rover_rewards = np.zeros((n_rovers, rover_steps))  # Keep track of rover rewards at each t
                for skill in range(n_suggestions):
                    # Get weights for CBA skill selector
                    for rover_id in range(n_rovers):
                        policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                        weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                        pops["CBA{0}".format(rover_id)].get_weights(weights)  # Suggestion Network Gets Weights

                    # Reset rovers to initial conditions
                    for rov in rd.rovers:
                        rd.rovers[rov].reset_rover()

                    chosen_pol = np.zeros(n_rovers)
                    for rov in rd.rovers:  # Initial rover scan of environment
                        rover_id = rd.rovers[rov].self_id
                        rd.rovers[rov].scan_environment(rd.rovers, rd.pois)
                        sensor_data = rd.rovers[rov].sensor_readings  # Unaltered sensor readings

                        # Select a skill using counterfactually shaped state information
                        target_pid = int(rover_skills[rover_id][skill])
                        suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, target_pid)
                        cba_input = np.sum((suggestion, sensor_data), axis=0)  # Shaped agent perception
                        pops["CBA{0}".format(rover_id)].get_inputs(cba_input)  # CBA network receives shaped input
                        cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()  # CBA picks skill
                        chosen_pol[rover_id] = int(cba_outputs)

                        # Rover uses selected skill
                        weights = rd.rovers[rov].policy_bank["Policy{0}".format(int(chosen_pol[rover_id]))]
                        rd.rovers[rov].get_weights(weights)
                        rd.rovers[rov].get_nn_outputs()

                    for step_id in range(rover_steps):
                        # Rover takes an action in the world
                        for rov in rd.rovers:
                            rd.rovers[rov].step(rd.world_x, rd.world_y)

                        # Rover scans environment and processes counterfactually shaped perceptions
                        for rov in rd.rovers:
                            rover_id = rd.rovers[rov].self_id
                            rd.rovers[rov].scan_environment(rd.rovers, rd.pois)
                            sensor_data = rd.rovers[rov].sensor_readings

                            # Select a skill using counterfactually shaped state information
                            target_pid = int(rover_skills[rover_id][skill])
                            suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, target_pid)
                            cba_input = np.sum((suggestion, sensor_data), axis=0)  # Shaped agent perception
                            pops["CBA{0}".format(rover_id)].get_inputs(cba_input)  # CBA network receives shaped input
                            cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()  # CBA picks skill
                            chosen_pol[rover_id] = int(cba_outputs)

                            # Rover uses selected skill
                            weights = rd.rovers[rov].policy_bank["Policy{0}".format(int(chosen_pol[rover_id]))]
                            rd.rovers[rov].get_weights(weights)
                            rd.rovers[rov].get_nn_outputs()

                        for poi in rd.pois:
                            rd.pois[poi].update_observer_distances(rd.rovers)

                        # Calculate Rewards
                        for rover_id in range(n_rovers):
                            reward = target_poi_reward(rover_id, rd.pois, int(rover_skills[rover_id][skill]))
                            rover_rewards[rover_id, step_id] = reward

                    # Update policy fitnesses
                    for rover_id in range(n_rovers):
                        policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                        pops["EA{0}".format(rover_id)].fitness[policy_id] += sum(rover_rewards[rover_id])/rover_steps

            # Choose parents and create new offspring population
            for rover_id in range(n_rovers):
                pops["EA{0}".format(rover_id)].down_select()

                # Record training performance data
                if gen % sample_rate == 0:
                    policy_rewards[rover_id].append(max(pops["EA{0}".format(rover_id)].fitness))

        # Record trial data
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)
            create_csv_file(policy_rewards[rover_id], 'Output_Data/Rover{0}'.format(rover_id), "CBA_Rewards.csv")

        srun += 1


def train_cba_custom_skills():
    """
    Train suggestions using a pre-trained playbook of rover policies
    """
    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]

    # Suggestion Parameters
    n_suggestions = p["n_suggestions"]
    s_inp = p["s_inputs"]
    s_hid = p["s_hidden"]
    s_out = p["s_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=s_inp, n_out=s_out, n_hid=s_hid)
        pops["CBA{0}".format(rover_id)] = CBANetwork()

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < stat_runs:
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for rover_id in range(n_rovers):
            pops["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population

        policy_rewards = [[] for i in range(n_rovers)]
        for gen in range(generations):
            # Create list of suggestions for rovers to use during training and reset rovers to initial positions
            for rover_id in range(n_rovers):
                pops["EA{0}".format(rover_id)].select_policy_teams()
                pops["EA{0}".format(rover_id)].reset_fitness()

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(population_size):
                rover_rewards = np.zeros((n_rovers, rover_steps))  # Keep track of rover rewards at each t
                for skill in range(n_suggestions):
                    # Get weights for CBA skill selector
                    for rover_id in range(n_rovers):
                        policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                        weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                        pops["CBA{0}".format(rover_id)].get_weights(weights)  # Suggestion Network Gets Weights

                    # Reset rovers to initial conditions
                    for rov in rd.rovers:
                        rd.rovers[rov].reset_rover()

                    for rov in rd.rovers:  # Initial rover scan of environment
                        rover_id = rd.rovers[rov].self_id
                        rd.rovers[rov].scan_environment(rd.rovers, rd.pois)
                        sensor_data = rd.rovers[rov].sensor_readings  # Unaltered sensor readings

                        # Select a skill using counterfactually shaped state information
                        target_pid = skill
                        suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, target_pid, sensor_data)
                        cba_input = np.sum((suggestion, sensor_data), axis=0)  # Shaped agent perception
                        pops["CBA{0}".format(rover_id)].get_inputs(cba_input)  # CBA network receives shaped input
                        cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()  # CBA picks skill
                        chosen_pol = int(cba_outputs)

                        # Rover uses selected skill
                        rx = rd.rovers[rov].x_pos
                        ry = rd.rovers[rov].y_pos
                        rd.rovers[rov].rover_actions = get_custom_action(chosen_pol, rd.pois, rx, ry)

                    for step_id in range(rover_steps):
                        # Rover takes an action in the world
                        for rov in rd.rovers:
                            rd.rovers[rov].custom_step(rd.world_x, rd.world_y)

                        # Rover scans environment and processes counterfactually shaped perceptions
                        for rov in rd.rovers:
                            rover_id = rd.rovers[rov].self_id
                            rd.rovers[rov].scan_environment(rd.rovers, rd.pois)
                            sensor_data = rd.rovers[rov].sensor_readings

                            # Select a skill using counterfactually shaped state information
                            target_pid = skill
                            suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, target_pid, sensor_data)
                            cba_input = np.sum((suggestion, sensor_data), axis=0)  # Shaped agent perception
                            pops["CBA{0}".format(rover_id)].get_inputs(cba_input)  # CBA network receives shaped input
                            cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()  # CBA picks skill
                            chosen_pol = int(cba_outputs)

                            # Rover uses selected skill
                            rx = rd.rovers[rov].x_pos
                            ry = rd.rovers[rov].y_pos
                            rd.rovers[rov].rover_actions = get_custom_action(chosen_pol, rd.pois, rx, ry)

                        for poi in rd.pois:
                            rd.pois[poi].update_observer_distances(rd.rovers)

                        # Calculate Rewards
                        for rover_id in range(n_rovers):
                            reward = target_poi_reward(rover_id, rd.pois, skill)
                            rover_rewards[rover_id, step_id] = reward

                    # Update policy fitnesses
                    for rover_id in range(n_rovers):
                        policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                        pops["EA{0}".format(rover_id)].fitness[policy_id] += sum(rover_rewards[rover_id])/rover_steps

            # Choose parents and create new offspring population
            for rover_id in range(n_rovers):
                pops["EA{0}".format(rover_id)].down_select()

                # Record training performance data
                if gen % sample_rate == 0:
                    policy_rewards[rover_id].append(max(pops["EA{0}".format(rover_id)].fitness))

        # Record trial data
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)
            create_csv_file(policy_rewards[rover_id], 'Output_Data/Rover{0}'.format(rover_id), "CBA_Rewards.csv")

        srun += 1


if __name__ == '__main__':
    """
    Train suggestions interpreter (must have already pre-trained agent playbook)
    """

    print("Training CBA Skill Selector")
    if p["custom_skills"]:
        train_cba_custom_skills()
    else:
        train_cba_learned_skills()
