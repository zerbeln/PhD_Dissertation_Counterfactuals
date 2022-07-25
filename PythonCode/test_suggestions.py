from cba_network import CBANetwork
from RoverDomain_Core.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
import numpy as np
from parameters import parameters as p
from itertools import product
from custom_rover_skills import get_custom_action
from global_functions import *
from cba import create_policy_bank, get_counterfactual_state


def find_best_suggestions(pbank_type, srun, c_list):
    """
    Test suggestions using the pre-trained policy bank
    """
    # Parameters
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Generate Hazard Areas (If Testing For Hazards)
    if p["active_hazards"]:
        for poi_id in p["hazardous_poi"]:
            rd.pois["P{0}".format(poi_id)].hazardous = True

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["CBA{0}".format(rover_id)] = CBANetwork()

    # Load Trained Suggestion Interpreter Weights
    for rover_id in range(n_rovers):
        if not p["custom_skills"]:
            rd.rovers["R{0}".format(rover_id)].policy_bank = create_policy_bank(pbank_type, rover_id, srun)
        s_weights = load_saved_policies('SelectionWeights{0}'.format(rover_id), rover_id, srun)
        pops["CBA{0}".format(rover_id)].get_weights(s_weights)

    best_rover_suggestion = None
    best_reward = None
    for c in c_list:
        sgst = [i for i in c]

        # Reset Rover
        for rov in rd.rovers:
            rd.rovers[rov].reset_rover()
        for rk in rd.rovers:  # Initial rover scan of environment
            rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
        for poi in rd.pois:
            rd.pois[poi].update_observer_distances(rd.rovers)

        # Create counterfactual for CBA
        for rov in rd.rovers:
            rover_id = rd.rovers[rov].self_id
            sensor_data = rd.rovers[rov].sensor_readings
            suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, sgst[rover_id], sensor_data)
            cba_input = np.sum((suggestion, sensor_data), axis=0)
            pops["CBA{0}".format(rover_id)].get_inputs(cba_input)

            # Determine action based on sensor inputs and suggestion
            cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()
            pol_id = int(cba_outputs)
            if p["custom_skills"]:
                rd.rovers[rov].rover_actions = get_custom_action(pol_id, rd.pois, rd.rovers[rov].x_pos, rd.rovers[rov].y_pos)
            else:
                weights = rd.rovers[rov].policy_bank["Policy{0}".format(pol_id)]
                rd.rovers[rov].get_weights(weights)
                rd.rovers[rov].get_nn_outputs()

        rewards = np.zeros(p["n_poi"])
        n_incursions = 0  # Number of times rovers violate a hazardous area
        for step_id in range(rover_steps):
            # Rover takes an action in the world
            for rov in rd.rovers:
                if p["custom_skills"]:
                    rd.rovers[rov].custom_step(rd.world_x, rd.world_y)
                else:
                    rd.rovers[rov].step(rd.world_x, rd.world_y)

            # Rover scans environment and processes suggestions
            for rov in rd.rovers:  # Rover scans environment
                rd.rovers[rov].scan_environment(rd.rovers, rd.pois)
            for poi in rd.pois:
                rd.pois[poi].update_observer_distances(rd.rovers)

            for rov in rd.rovers:
                rover_id = rd.rovers[rov].self_id
                sensor_data = rd.rovers[rov].sensor_readings
                suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, sgst[rover_id], sensor_data)
                cba_input = np.sum((suggestion, sensor_data), axis=0)
                pops["CBA{0}".format(rover_id)].get_inputs(cba_input)

                # Determine action based on sensor inputs and suggestion
                cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()
                pol_id = int(cba_outputs)
                if p["custom_skills"]:
                    rd.rovers[rov].rover_actions = get_custom_action(pol_id, rd.pois, rd.rovers[rov].x_pos, rd.rovers[rov].y_pos)
                else:
                    weights = rd.rovers[rov].policy_bank["Policy{0}".format(pol_id)]
                    rd.rovers[rov].get_weights(weights)
                    rd.rovers[rov].get_nn_outputs()

            # Calculate Global Reward
            poi_rewards = rd.calc_global()
            for poi_id in range(p["n_poi"]):
                if rd.pois["P{0}".format(poi_id)].hazardous and poi_rewards[poi_id] < 0:
                    n_incursions += 1
                    poi_rewards[poi_id] = -10.0 * n_incursions
                elif poi_rewards[poi_id] > rewards[poi_id] and not rd.pois["P{0}".format(poi_id)].hazardous:
                    rewards[poi_id] = poi_rewards[poi_id]

        if best_reward is None or sum(rewards) > best_reward:
            best_reward = sum(rewards)
            best_rover_suggestion = sgst

    create_csv_file(best_rover_suggestion, "Output_Data/", "BestRoverCounterfactuals.csv")
    return best_rover_suggestion


def test_cba_custom_skills(counterfactuals):
    """
    Test suggestions using the hand created policy bank
    """
    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Generate Hazard Areas (If Testing For Hazards)
    if p["active_hazards"]:
        for poi_id in p["hazardous_poi"]:
            rd.pois["P{0}".format(poi_id)].hazardous = True

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["CBA{0}".format(rover_id)] = CBANetwork()

    average_reward = 0
    reward_history = []  # Keep track of team performance throughout training
    incursion_tracker = []
    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    srun = p["starting_srun"]
    while srun < stat_runs:  # Perform statistical runs
        sgst = counterfactuals["S{0}".format(srun)]

        # Load Trained Suggestion Interpreter Weights
        for rover_id in range(n_rovers):
            s_weights = load_saved_policies('SelectionWeights{0}'.format(rover_id), rover_id, srun)
            pops["CBA{0}".format(rover_id)].get_weights(s_weights)

        # Reset Rover
        for rov in rd.rovers:
            rd.rovers[rov].reset_rover()
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 0] = rd.rovers[rov].x_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 1] = rd.rovers[rov].y_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 2] = rd.rovers[rov].theta_pos

        for rk in rd.rovers:  # Initial rover scan of environment
            rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
        for poi in rd.pois:
            rd.pois[poi].update_observer_distances(rd.rovers)

        # Create counterfactual for CBA
        for rov in rd.rovers:
            rover_id = rd.rovers[rov].self_id
            sensor_data = rd.rovers[rov].sensor_readings
            suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, sgst[rover_id], sensor_data)
            cba_input = np.sum((suggestion, sensor_data), axis=0)
            pops["CBA{0}".format(rover_id)].get_inputs(cba_input)

            # Determine action based on sensor inputs and suggestion
            cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()
            pol_id = int(cba_outputs)
            rd.rovers[rov].rover_actions = get_custom_action(pol_id, rd.pois, rd.rovers[rov].x_pos, rd.rovers[rov].y_pos)
            rd.rovers[rov].custom_step(rd.world_x, rd.world_y)

        rewards = np.zeros(p["n_poi"])
        n_incursions = 0
        for step_id in range(rover_steps):
            # Rover takes an action in the world
            for rov in rd.rovers:
                rd.rovers[rov].custom_step(rd.world_x, rd.world_y)
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 0] = rd.rovers[rov].x_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 1] = rd.rovers[rov].y_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 2] = rd.rovers[rov].theta_pos

            # Rover scans environment and processes suggestions
            for rk in rd.rovers:  # Rover scans environment
                rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
            for poi in rd.pois:
                rd.pois[poi].update_observer_distances(rd.rovers)

            for rov in rd.rovers:
                rover_id = rd.rovers[rov].self_id
                sensor_data = rd.rovers[rov].sensor_readings
                suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, sgst[rover_id], sensor_data)
                cba_input = np.sum((suggestion, sensor_data), axis=0)
                pops["CBA{0}".format(rover_id)].get_inputs(cba_input)

                # Determine action based on sensor inputs and suggestion
                cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()
                pol_id = int(cba_outputs)
                rx = rd.rovers[rov].x_pos
                ry = rd.rovers[rov].y_pos
                rd.rovers[rov].rover_actions = get_custom_action(pol_id, rd.pois, rx, ry)
                rd.rovers[rov].custom_step(rd.world_x, rd.world_y)

            # Calculate Global Reward
            poi_rewards = rd.calc_global()
            for poi_id in range(p["n_poi"]):
                if rd.pois["P{0}".format(poi_id)].hazardous and poi_rewards[poi_id] < 0:
                    n_incursions += 1
                    rewards[poi_id] = -10.0 * n_incursions
                elif poi_rewards[poi_id] > rewards[poi_id] and not rd.pois["P{0}".format(poi_id)].hazardous:
                    rewards[poi_id] = poi_rewards[poi_id]

        reward_history.append(sum(rewards))
        incursion_tracker.append(n_incursions)
        average_reward += sum(rewards)
        srun += 1

    print(average_reward/stat_runs)
    create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths")
    create_csv_file(reward_history, "Output_Data/", "Final_GlobalRewards.csv")
    create_csv_file(incursion_tracker, "Output_Data/", "HazardIncursions.csv")
    if p["vis_running"]:
        run_visualizer()


def test_cba(pbank_type, counterfactuals):
    """
    Test suggestions using the pre-trained policy bank
    """
    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Generate Hazard Areas (If Testing For Hazards)
    if p["active_hazards"]:
        for poi_id in p["hazardous_poi"]:
            rd.pois["P{0}".format(poi_id)].hazardous = True

    # Create dictionary for each instance of rover and corresponding NN and EA population
    pops = {}
    for rover_id in range(n_rovers):
        pops["CBA{0}".format(rover_id)] = CBANetwork()

    average_reward = 0
    reward_history = []  # Keep track of team performance throughout training
    incursion_tracker = []
    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    srun = p["starting_srun"]
    while srun < stat_runs:  # Perform statistical runs
        sgst = counterfactuals["S{0}".format(srun)]
        # Load Trained Suggestion Interpreter Weights
        for rover_id in range(n_rovers):
            rd.rovers["R{0}".format(rover_id)].policy_bank = create_policy_bank(pbank_type, rover_id, srun)
            s_weights = load_saved_policies('SelectionWeights{0}'.format(rover_id), rover_id, srun)
            pops["CBA{0}".format(rover_id)].get_weights(s_weights)

        # Reset Rover
        for rov in rd.rovers:
            rd.rovers[rov].reset_rover()
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 0] = rd.rovers[rov].x_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 1] = rd.rovers[rov].y_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 2] = rd.rovers[rov].theta_pos

        for rk in rd.rovers:  # Initial rover scan of environment
            rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
        for poi in rd.pois:
            rd.pois[poi].update_observer_distances(rd.rovers)

        # Create counterfactual for CBA
        for rov in rd.rovers:
            rover_id = rd.rovers[rov].self_id
            sensor_data = rd.rovers[rov].sensor_readings
            suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, sgst[rover_id], sensor_data)
            cba_input = np.sum((suggestion, sensor_data), axis=0)
            pops["CBA{0}".format(rover_id)].get_inputs(cba_input)

            # Determine action based on sensor inputs and suggestion
            cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()
            pol_id = int(cba_outputs)
            weights = rd.rovers[rov].policy_bank["Policy{0}".format(pol_id)]
            rd.rovers[rov].get_weights(weights)
            rd.rovers[rov].get_nn_outputs()

        rewards = np.zeros(p["n_poi"])
        n_incursions = 0
        for step_id in range(rover_steps):
            # Rover takes an action in the world
            for rov in rd.rovers:
                rd.rovers[rov].step(rd.world_x, rd.world_y)
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 0] = rd.rovers[rov].x_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 1] = rd.rovers[rov].y_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 2] = rd.rovers[rov].theta_pos

            # Rover scans environment and processes suggestions
            for rk in rd.rovers:  # Rover scans environment
                rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
            for poi in rd.pois:
                rd.pois[poi].update_observer_distances(rd.rovers)

            for rov in rd.rovers:
                rover_id = rd.rovers[rov].self_id
                sensor_data = rd.rovers[rov].sensor_readings
                suggestion = get_counterfactual_state(rd.pois, rd.rovers, rover_id, sgst[rover_id], sensor_data)
                cba_input = np.sum((suggestion, sensor_data), axis=0)
                pops["CBA{0}".format(rover_id)].get_inputs(cba_input)

                # Determine action based on sensor inputs and suggestion
                cba_outputs = pops["CBA{0}".format(rover_id)].get_outputs()
                pol_id = int(cba_outputs)
                weights = rd.rovers[rov].policy_bank["Policy{0}".format(pol_id)]
                rd.rovers[rov].get_weights(weights)
                rd.rovers[rov].get_nn_outputs()

            # Calculate Global Reward
            poi_rewards = rd.calc_global()
            for poi_id in range(p["n_poi"]):
                if rd.pois["P{0}".format(poi_id)].hazardous and poi_rewards[poi_id] < 0:
                    n_incursions += 1
                    rewards[poi_id] = -10.0 * n_incursions
                elif poi_rewards[poi_id] > rewards[poi_id] and not rd.pois["P{0}".format(poi_id)].hazardous:
                    rewards[poi_id] = poi_rewards[poi_id]

        reward_history.append(sum(rewards))
        incursion_tracker.append(n_incursions)
        average_reward += sum(rewards)
        srun += 1

    average_reward /= stat_runs
    print(average_reward)
    create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths")
    create_csv_file(reward_history, "Output_Data/", "Final_GlobalRewards.csv")
    create_csv_file(incursion_tracker, "Output_Data/", "HazardIncursions.csv")
    if p["vis_running"]:
        run_visualizer()


if __name__ == '__main__':
    # Test Performance of CBA
    counterfactuals = {}
    if p["suggestion_type"] == "Best_Total":
        choices = range(p["n_suggestions"])
        n = p["n_rovers"]
        t_list = [choices] * n
        for srun in range(p["stat_runs"]):
            print(srun+1, "/", p["stat_runs"])
            c_list = (product(*t_list))
            counterfactuals["S{0}".format(srun)] = find_best_suggestions(p["skill_type"], srun, c_list)
    elif p["suggestion_type"] == "Best_Random":
        c_list = np.random.randint(0, p["n_suggestions"], (p["c_list_size"], p["n_rovers"]))
        for srun in range(p["stat_runs"]):
            print(srun+1, "/", p["stat_runs"])
            counterfactuals["S{0}".format(srun)] = find_best_suggestions(p["skill_type"], srun, c_list)
    else:  # Custom
        rover_suggestions = [0, 0, 0]
        for srun in range(p["stat_runs"]):
            counterfactuals["S{0}".format(srun)] = rover_suggestions

    # Testing CBA for rover using hand created policies or learned policies
    if p["custom_skills"]:
        test_cba_custom_skills(counterfactuals)
    else:
        test_cba(p["skill_type"], counterfactuals)

