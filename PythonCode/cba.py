from global_functions import *
from ccea import Ccea
from cba_network import CBANetwork
from RoverDomain_Core.rover_domain import RoverDomain
from RewardFunctions.cba_rewards import *
from custom_rover_skills import get_custom_action
from parameters import parameters as p
import random
import numpy as np


def create_policy_bank(playbook_type, rover_id, srun):
    """
    Load the pre-trained rover skills into a dictionary from saved pickle files
    """
    policy_bank = {}

    if playbook_type == "Target_Quadrant":
        for q_id in range(4):
            w = load_saved_policies("TowardQuadrant{0}".format(q_id), rover_id, srun)
            policy_bank["Policy{0}".format(q_id)] = w
    elif playbook_type == "Target_POI":
        for poi_id in range(p["n_poi"]):
            w = load_saved_policies("TowardPOI{0}".format(poi_id), rover_id, srun)
            policy_bank["Policy{0}".format(poi_id)] = w

    return policy_bank


def get_counterfactual_state(pois, rovers, rover_id, suggestion, sensor_data):
    """
    Create a counteractual state input to represent agent suggestions
    """
    n_brackets = int(360.0 / p["angle_res"])
    counterfactual_state = np.zeros(int(n_brackets * 2))

    # If suggestion is to go towards a POI -> create a counterfactual otherwise, use no counterfactual
    if suggestion < p["n_poi"]:
        rx = rovers["R{0}".format(rover_id)].loc[0]
        ry = rovers["R{0}".format(rover_id)].loc[1]
        cfact_poi = create_counterfactual_poi_state(pois, rx, ry, n_brackets, suggestion, sensor_data)
        cfact_rover = create_counterfactual_rover_state(pois, rovers, rx, ry, n_brackets, rover_id, suggestion, sensor_data)

        for i in range(n_brackets):
            counterfactual_state[i] = cfact_poi[i]
            counterfactual_state[n_brackets + i] = cfact_rover[i]

    # Possible idea: make counterfactual_state counteract actual state if suggestion is to stop

    return counterfactual_state


def create_counterfactual_poi_state(pois, rx, ry, n_brackets, suggestion, sensor_data):
    """
    Construct a counterfactual state based on POI sensors
    """
    c_poi_state = np.zeros(n_brackets)
    poi_quadrant = pois["P{0}".format(suggestion)].quadrant
    dist = get_squared_dist(pois["P{0}".format(suggestion)].loc[0], pois["P{0}".format(suggestion)].loc[1], rx, ry)

    for bracket in range(n_brackets):
        if bracket == poi_quadrant:
            c_poi_state[bracket] = (pois["P{0}".format(suggestion)].value/dist) - sensor_data[bracket]
        else:
            c_poi_state[bracket] = -(1 + sensor_data[bracket])

    return c_poi_state


def create_counterfactual_rover_state(pois, rovers, rx, ry, n_brackets, rover_id, suggestion, sensor_data):
    """
    Construct a counterfactual state input based on Rover sensors
    """
    rover_state = np.zeros(n_brackets)
    poi_quadrant = pois["P{0}".format(suggestion)].quadrant
    for bracket in range(n_brackets):
        if bracket == poi_quadrant:
            rover_state[bracket] = -(1 + sensor_data[bracket+4])
        else:
            if bracket == 1:
                crx = (1/4)*p["x_dim"]
                cry = (1/4)*p["y_dim"]
            elif bracket == 0:
                crx = (3/4)*p["x_dim"]
                cry = (1/4)*p["y_dim"]
            elif bracket == 2:
                crx = (1/4)*p["x_dim"]
                cry = (3/4)*p["y_dim"]
            elif bracket == 3:
                crx = (3/4)*p["x_dim"]
                cry = (3/4)*p["y_dim"]

            dist = get_squared_dist(crx, cry, rx, ry)
            rover_state[bracket] = (2/dist) - sensor_data[bracket+4]

    return rover_state

def train_cba_learned_skills():
    """
    Train suggestions using a pre-trained playbook of rover policies
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(p["n_rovers"]):
        pops["EA{0}".format(rover_id)] = Ccea(p["pop_size"], n_inp=p["s_inp"], n_out=p["s_out"], n_hid=p["s_hid"])
        pops["CBA{0}".format(rover_id)] = CBANetwork()

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop and Load Pre-Trained Policies
        for rover_id in range(p["n_rovers"]):
            pops["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population
            rd.rovers["R{0}".format(rover_id)].policy_bank = create_policy_bank(p["skill_type"], rover_id, srun)

        policy_rewards = [[] for i in range(p["n_rovers"])]
        for gen in range(p["generations"]):
            # Create list of suggestions for rovers to use during training and reset rovers to initial positions
            rover_skills = []
            for rover_id in range(p["n_rovers"]):
                pops["EA{0}".format(rover_id)].select_policy_teams()
                pops["EA{0}".format(rover_id)].reset_fitness()
                skill_sample = random.sample(range(p["n_suggestions"]), p["n_suggestions"])
                rover_skills.append(skill_sample)

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(p["pop_size"]):
                rover_rewards = np.zeros((p["n_rovers"], p["steps"]))  # Keep track of rover rewards at each t
                for skill in range(p["n_suggestions"]):
                    # Get weights for CBA skill selector
                    for rover_id in range(p["n_rovers"]):
                        policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                        weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                        pops["CBA{0}".format(rover_id)].get_weights(weights)  # Suggestion Network Gets Weights

                    # Reset rovers to initial conditions
                    for rov in rd.rovers:
                        rd.rovers[rov].reset_rover()

                    chosen_pol = np.zeros(p["n_rovers"])
                    for rov in rd.rovers:  # Initial rover scan of environment
                        rover_id = rd.rovers[rov].self_id
                        rd.rovers[rov].scan_environment(rd.rovers, rd.pois)
                        sensor_data = rd.rovers[rov].sensor_readings  # Unaltered sensor readings

                        # Select a skill using counterfactually shaped state information
                        target_pid = int(rover_skills[rover_id][skill])
                        suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, target_pid, sensor_data)
                        cba_input = np.sum((suggestion, sensor_data), axis=0)  # Shaped agent perception
                        pops["CBA{0}".format(rover_id)].get_inputs(cba_input)  # CBA network receives shaped input
                        cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()  # CBA picks skill
                        chosen_pol[rover_id] = int(cba_outputs)

                        # Rover uses selected skill
                        weights = rd.rovers[rov].policy_bank["Policy{0}".format(int(chosen_pol[rover_id]))]
                        rd.rovers[rov].get_weights(weights)
                        rd.rovers[rov].get_nn_outputs()

                    for step_id in range(p["steps"]):
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
                            suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, target_pid, sensor_data)
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
                        for rover_id in range(p["n_rovers"]):
                            reward = target_poi_reward(rover_id, rd.pois, int(rover_skills[rover_id][skill]))
                            rover_rewards[rover_id, step_id] = reward

                    # Update policy fitnesses
                    for rover_id in range(p["n_rovers"]):
                        policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                        pops["EA{0}".format(rover_id)].fitness[policy_id] += sum(rover_rewards[rover_id])/p["steps"]

            # Choose parents and create new offspring population
            for rover_id in range(p["n_rovers"]):
                pops["EA{0}".format(rover_id)].down_select()

                # Record training performance data
                if gen % p["sample_rate"] == 0:
                    policy_rewards[rover_id].append(max(pops["EA{0}".format(rover_id)].fitness))

        # Record trial data
        for rover_id in range(p["n_rovers"]):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)
            create_csv_file(policy_rewards[rover_id], 'Output_Data/Rover{0}'.format(rover_id), "CBA_Rewards.csv")

        srun += 1


def train_cba_custom_skills():
    """
    Train suggestions using a pre-trained playbook of rover policies
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(p["n_rovers"]):
        pops["EA{0}".format(rover_id)] = Ccea(p["pop_size"], n_inp=p["s_inp"], n_out=p["s_out"], n_hid=p["s_hid"])
        pops["CBA{0}".format(rover_id)] = CBANetwork()

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for rover_id in range(p["n_rovers"]):
            pops["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population

        policy_rewards = [[] for i in range(p["n_rovers"])]
        for gen in range(p["generations"]):
            # Create list of suggestions for rovers to use during training and reset rovers to initial positions
            for rover_id in range(p["n_rovers"]):
                pops["EA{0}".format(rover_id)].select_policy_teams()
                pops["EA{0}".format(rover_id)].reset_fitness()

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(p["pop_size"]):
                rover_rewards = np.zeros((p["n_rovers"], p["steps"]))  # Keep track of rover rewards at each t
                for skill in range(p["n_suggestions"]):
                    # Get weights for CBA skill selector
                    for rover_id in range(p["n_rovers"]):
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

                    for step_id in range(p["steps"]):
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
                        for rover_id in range(p["n_rovers"]):
                            reward = target_poi_reward(rover_id, rd.pois, skill)
                            rover_rewards[rover_id, step_id] = reward

                    # Update policy fitnesses
                    for rover_id in range(p["n_rovers"]):
                        policy_id = int(pops["EA{0}".format(rover_id)].team_selection[team_number])
                        pops["EA{0}".format(rover_id)].fitness[policy_id] += sum(rover_rewards[rover_id])/p["steps"]

            # Choose parents and create new offspring population
            for rover_id in range(p["n_rovers"]):
                pops["EA{0}".format(rover_id)].down_select()

                # Record training performance data
                if gen % p["sample_rate"] == 0:
                    policy_rewards[rover_id].append(max(pops["EA{0}".format(rover_id)].fitness))

        # Record trial data
        for rover_id in range(p["n_rovers"]):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)
            create_csv_file(policy_rewards[rover_id], 'Output_Data/Rover{0}'.format(rover_id), "CBA_Rewards.csv")

        srun += 1
