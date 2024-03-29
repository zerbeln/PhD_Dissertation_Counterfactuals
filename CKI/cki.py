from global_functions import *
from EvolutionaryAlgorithms.ccea import CCEA
from NeuralNetworks.neural_network import NeuralNetwork
from RoverDomainCore.rover_domain import RoverDomain
from CKI.cki_rewards import *
from CKI.custom_rover_skills import get_custom_action
from parameters import parameters as p
import numpy as np
import random


def calculate_poi_sectors(pois):
    """
    Calculate which quadrant (or sector) of the environment each POI exists in
    """
    for poi in pois:
        angle = get_angle(pois[poi].loc[0], pois[poi].loc[1], p["x_dim"]/2, p["y_dim"]/2)

        sector = int(angle/p["angle_res"])
        pois[poi].quadrant = sector


def create_policy_bank(playbook_type, rover_id, srun):
    """
    Load the pre-trained rover skills into a dictionary from saved pickle files
    """
    policy_bank = {}

    if playbook_type == "Target_Quadrant":
        for q_id in range(4):
            w = load_saved_policies(f'TowardQuadrant{q_id}', rover_id, srun)
            policy_bank[f'Policy{q_id}'] = w
    elif playbook_type == "Target_POI":
        for poi_id in range(p["n_poi"]):
            w = load_saved_policies(f'TowardPOI{poi_id}', rover_id, srun)
            policy_bank[f'Policy{poi_id}'] = w

    return policy_bank


def get_counterfactual_state(pois, rovers, rover_id, c_state, sensor_data):
    """
    Create a counteractual state input to shape agent perceptions
    """
    n_brackets = int(360.0 / p["angle_res"])
    counterfactual_state = np.zeros(int(n_brackets * 2))

    # If c_state is to go towards a POI -> create a counterfactual otherwise, use negative counterfactual state
    if c_state < p["n_poi"]:
        rx = rovers[f'R{rover_id}'].loc[0]
        ry = rovers[f'R{rover_id}'].loc[1]
        cfact_poi = create_counterfactual_poi_state(pois, rx, ry, n_brackets, c_state, sensor_data)
        cfact_rover = create_counterfactual_rover_state(pois, rx, ry, n_brackets, c_state, sensor_data)

        for i in range(n_brackets):
            counterfactual_state[i] = cfact_poi[i]
            counterfactual_state[n_brackets + i] = cfact_rover[i]
    else:
        counterfactual_state = np.ones(int(n_brackets * 2)) * -1

    return counterfactual_state


def create_counterfactual_poi_state(pois, rx, ry, n_brackets, c_state, sensor_data):
    """
    Construct a counterfactual state based on POI sensors
    """
    c_poi_state = np.zeros(n_brackets)
    poi_quadrant = pois[f'P{c_state}'].quadrant
    dist = get_squared_dist(pois[f'P{c_state}'].loc[0], pois[f'P{c_state}'].loc[1], rx, ry)

    for bracket in range(n_brackets):
        if bracket == poi_quadrant:
            c_poi_state[bracket] = 2*pois[f'P{c_state}'].value/dist  # Adding in two counterfactual POI
        else:
            c_poi_state[bracket] = -(1 + sensor_data[bracket])  # No POI in other quadrants

    return c_poi_state


def create_counterfactual_rover_state(pois, rx, ry, n_brackets, c_state, sensor_data):
    """
    Construct a counterfactual state input based on Rover sensors
    """
    rover_state = np.zeros(n_brackets)
    poi_quadrant = pois[f'P{c_state}'].quadrant
    for bracket in range(n_brackets):
        if bracket == poi_quadrant:
            rover_state[bracket] = -(1 + sensor_data[bracket+4])  # No rovers in target quadrant
        else:
            # Add some slight "noise" to counterfactual rover positions
            if bracket == 1:
                crx = (1/4)*p["x_dim"] + random.uniform(-2, 2)
                cry = (1/4)*p["y_dim"] + random.uniform(-2, 2)
            elif bracket == 0:
                crx = (3/4)*p["x_dim"] + random.uniform(-2, 2)
                cry = (1/4)*p["y_dim"] + random.uniform(-2, 2)
            elif bracket == 2:
                crx = (1/4)*p["x_dim"] + random.uniform(-2, 2)
                cry = (3/4)*p["y_dim"] + random.uniform(-2, 2)
            elif bracket == 3:
                crx = (3/4)*p["x_dim"] + random.uniform(-2, 2)
                cry = (3/4)*p["y_dim"] + random.uniform(-2, 2)

            dist = get_squared_dist(crx, cry, rx, ry)
            rover_state[bracket] = (2/dist) - sensor_data[bracket+4]

    return rover_state


def cba_step(rovers, pois, rover_actions):
    # Rovers take action from neural network
    for rv in rovers:
        dx = 2 * p["dmax"] * (rover_actions[rovers[rv].rover_id][0] - 0.5)
        dy = 2 * p["dmax"] * (rover_actions[rovers[rv].rover_id][1] - 0.5)

        # Calculate new rover X Position
        x = dx + rovers[rv].loc[0]

        # Rovers cannot move beyond boundaries of the world
        if x < 0:
            x = 0
        elif x > p["x_dim"] - 1:
            x = p["x_dim"] - 1

        # Calculate new rover Y Position
        y = dy + rovers[rv].loc[1]

        # Rovers cannot move beyond boundaries of the world
        if y < 0:
            y = 0
        elif y > p["y_dim"] - 1:
            y = p["y_dim"] - 1

        # Update rover position
        rovers[rv].loc[0] = x
        rovers[rv].loc[1] = y

    # Update distance tracker for POIs and Rovers
    for poi in pois:
        pois[poi].update_observer_distances(rovers)


def train_cki():
    """
    Train CKI rovers using a hand-crafted set of rover skills
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()
    calculate_poi_sectors(rd.pois)

    # Create dictionaries for rover CCEA populations and neural networks
    pops = {}
    networks = {}
    for rover_id in range(p["n_rovers"]):
        pops[f'EA{rover_id}'] = CCEA(n_inp=p["cki_inp"], n_hid=p["cki_hid"], n_out=p["cki_out"])
        networks[f'NN{rover_id}'] = NeuralNetwork(n_inp=p["cki_inp"], n_hid=p["cki_hid"], n_out=p["cki_out"])

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        print("Run: %i" % srun)

        # Reset Rover and create new CCEA pops
        for rover_id in range(p["n_rovers"]):
            pops[f'EA{rover_id}'].create_new_population()  # Create new CCEA population

        policy_rewards = [[] for i in range(p["n_rovers"])]
        for gen in range(p["generations"]):
            # Reset fitness vector and select CCEA teams
            for rover_id in range(p["n_rovers"]):
                pops[f'EA{rover_id}'].select_policy_teams()
                pops[f'EA{rover_id}'].reset_fitness()

            # Each policy in CCEA is tested in randomly selected teams
            for team_number in range(p["pop_size"]):
                rover_rewards = np.zeros((p["n_rovers"], p["steps"]))  # Keep track of rover rewards at each t

                # For each skill a rover possesses, test counterfactual states targeting that skill
                for skill in range(p["n_skills"]):
                    # Reset environment to initial conditions and select network weights
                    rd.reset_world(0)
                    for rover_id in range(p["n_rovers"]):
                        policy_id = int(pops[f'EA{rover_id}'].team_selection[team_number])
                        weights = pops[f'EA{rover_id}'].population[f'pol{policy_id}']
                        networks[f'NN{rover_id}'].get_weights(weights)  # CKI Network Gets Weights

                    for step_id in range(p["steps"]):
                        # Rover scans environment and processes counterfactually shaped perceptions
                        rover_actions = []
                        for rv in rd.rovers:
                            rover_id = rd.rovers[rv].rover_id
                            rd.rovers[rv].scan_environment(rd.rovers, rd.pois)
                            sensor_data = rd.rovers[rv].observations  # Unaltered sensor readings

                            # Select a skill using counterfactually shaped state information
                            c_sensor_data = get_counterfactual_state(rd.pois, rd.rovers, rover_id, skill, sensor_data)
                            cba_input = np.sum((c_sensor_data, sensor_data), axis=0)  # Shaped agent perception
                            cba_outputs = networks[f'NN{rover_id}'].run_rover_nn(cba_input)  # CKI picks skill
                            chosen_pol = int(np.argmax(cba_outputs))

                            # Store rover action output
                            action = get_custom_action(chosen_pol, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                            rover_actions.append(action)

                        cba_step(rd.rovers, rd.pois, rover_actions)

                        # Calculate Rewards
                        for rover_id in range(p["n_rovers"]):
                            reward = target_poi_reward(rover_id, rd.pois, skill, rover_actions)
                            rover_rewards[rover_id, step_id] = reward

                    # Update policy fitnesses
                    for rover_id in range(p["n_rovers"]):
                        policy_id = int(pops[f'EA{rover_id}'].team_selection[team_number])
                        pops[f'EA{rover_id}'].fitness[policy_id] += sum(rover_rewards[rover_id])/p["steps"]

            # Choose parents and create new offspring population
            for rover_id in range(p["n_rovers"]):
                pops[f'EA{rover_id}'].down_select()

                # Record training performance data
                if gen % p["sample_rate"] == 0:
                    policy_rewards[rover_id].append(max(pops[f'EA{rover_id}'].fitness))

        # Record trial data
        for rover_id in range(p["n_rovers"]):
            best_policy_id = np.argmax(pops[f'EA{rover_id}'].fitness)
            weights = pops[f'EA{rover_id}'].population[f'pol{best_policy_id}']
            save_best_policies(weights, srun, f'SelectionWeights{rover_id}', rover_id)
            create_csv_file(policy_rewards[rover_id], f'Output_Data/Rover{rover_id}', "CKI_Rewards.csv")

        srun += 1
