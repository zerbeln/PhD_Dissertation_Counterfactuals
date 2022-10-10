from rover_neural_network import NeuralNetwork
from RoverDomain_Core.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
import numpy as np
from parameters import parameters as p
from itertools import product
from CBA.custom_rover_skills import get_custom_action
from global_functions import *
from CBA.cba import get_counterfactual_state, calculate_poi_sectors


def find_best_suggestions(srun, c_list):
    """
    Test suggestions using the pre-trained policy bank
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()
    calculate_poi_sectors(rd.pois)

    # Generate Hazard Areas (If Testing For Hazards)
    if p["active_hazards"]:
        for poi_id in p["hazardous_poi"]:
            rd.pois["P{0}".format(poi_id)].hazardous = True

    # Create dictionary for each instance of rover and corresponding NN and EA population
    networks = {}
    rover_skill_selections = {}
    for rover_id in range(p["n_rovers"]):
        networks["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["s_inp"], n_hid=p["s_hid"], n_out=p["s_out"])
        rover_skill_selections["RV{0}".format(rover_id)] = [[0 for i in range(p["n_skills"])] for j in range(p["n_skills"])]

    # Load Trained Suggestion Interpreter Weights
    for rover_id in range(p["n_rovers"]):
        s_weights = load_saved_policies('SelectionWeights{0}'.format(rover_id), rover_id, srun)
        networks["NN{0}".format(rover_id)].get_weights(s_weights)

    best_rover_suggestion = None
    best_reward = None

    for c in c_list:
        sgst = [i for i in c]

        # Reset environment to initial conditions
        rd.reset_world()

        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        n_incursions = 0  # Number of times rovers violate a hazardous area
        for step_id in range(p["steps"]):
            rover_actions = []
            for rv in rd.rovers:
                rover_id = rd.rovers[rv].rover_id
                rd.rovers[rv].scan_environment(rd.rovers, rd.pois)
                sensor_data = rd.rovers[rv].observations  # Unaltered sensor readings

                # Select a skill using counterfactually shaped state information
                c_sensor_data = get_counterfactual_state(rd.pois, rd.rovers, rover_id, sgst[rover_id], sensor_data)
                cba_input = np.sum((c_sensor_data, sensor_data), axis=0)  # Shaped agent perception
                cba_outputs = networks["NN{0}".format(rover_id)].run_rover_nn(cba_input)  # CBA picks skill
                chosen_pol = int(np.argmax(cba_outputs))
                rover_skill_selections["RV{0}".format(rover_id)][sgst[rover_id]][chosen_pol] += 1

                # Determine action based on sensor inputs and suggestion
                action = get_custom_action(chosen_pol, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                rover_actions.append(action)

            # Environment takes in rover actions and returns next state and global reward
            step_rewards = rd.step(rover_actions)
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]
                if rd.pois["P{0}".format(poi_id)].hazardous:
                    for dist in rd.pois["P{0}".format(poi_id)].observer_distances:
                        if dist < p["observation_radius"]:
                            n_incursions += 1

        # Calculate episodic global reward
        g_reward = 0
        for p_reward in poi_rewards:
            g_reward += max(p_reward)
        g_reward -= (n_incursions*10)
        if best_reward is None or g_reward > best_reward:
            best_reward = g_reward
            best_rover_suggestion = sgst

    create_csv_file(best_rover_suggestion, "Output_Data/", "BestRoverCounterfactuals.csv")

    return best_rover_suggestion, rover_skill_selections


def test_cba(counterfactuals):
    """
    Test suggestions using the hand created policy bank
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()
    calculate_poi_sectors(rd.pois)

    # Generate Hazard Areas (If Testing For Hazards)
    if p["active_hazards"]:
        for poi_id in p["hazardous_poi"]:
            rd.pois["P{0}".format(poi_id)].hazardous = True

    # Create dictionary for each instance of rover and corresponding NN and EA population
    networks = {}
    for rover_id in range(p["n_rovers"]):
        networks["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["s_inp"], n_hid=p["s_hid"], n_out=p["s_out"])

    average_reward = 0
    reward_history = []  # Keep track of team performance throughout training
    incursion_tracker = []  # Keep track of the number of hazard area violations each stat run
    final_rover_path = np.zeros((p["stat_runs"], p["n_rovers"], p["steps"], 3))

    srun = p["starting_srun"]
    while srun < p["stat_runs"]:  # Perform statistical runs
        sgst = counterfactuals["S{0}".format(srun)]

        # Load Trained Suggestion Interpreter Weights
        for rover_id in range(p["n_rovers"]):
            s_weights = load_saved_policies('SelectionWeights{0}'.format(rover_id), rover_id, srun)
            networks["NN{0}".format(rover_id)].get_weights(s_weights)

        # Reset environment to initial conditions
        rd.reset_world()
        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        n_incursions = 0
        for step_id in range(p["steps"]):
            rover_actions = []
            for rv in rd.rovers:
                # Record rover paths
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 0] = rd.rovers[rv].loc[0]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 1] = rd.rovers[rv].loc[1]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 2] = rd.rovers[rv].loc[2]

                # Rovers observe environment
                rover_id = rd.rovers[rv].rover_id
                rd.rovers[rv].scan_environment(rd.rovers, rd.pois)
                sensor_data = rd.rovers[rv].observations  # Unaltered sensor readings

                # Select a skill using counterfactually shaped state information
                c_sensor_data = get_counterfactual_state(rd.pois, rd.rovers, rover_id, sgst[rover_id], sensor_data)
                cba_input = np.sum((c_sensor_data, sensor_data), axis=0)  # Shaped agent perception
                cba_outputs = networks["NN{0}".format(rover_id)].run_rover_nn(cba_input)  # CBA picks skill
                chosen_pol = int(np.argmax(cba_outputs))

                # Determine action based on sensor inputs and suggestion
                action = get_custom_action(chosen_pol, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                rover_actions.append(action)

            # Environment takes in rover actions and returns next state and global reward
            step_rewards = rd.step(rover_actions)
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]
                if rd.pois["P{0}".format(poi_id)].hazardous:
                    for dist in rd.pois["P{0}".format(poi_id)].observer_distances:
                        if dist < p["observation_radius"]:
                            n_incursions += 1

        # Calculate episodic global reward
        g_reward = 0
        for p_reward in poi_rewards:
            g_reward += max(p_reward)
        g_reward -= (n_incursions*10)
        reward_history.append(g_reward)
        incursion_tracker.append(n_incursions)
        average_reward += g_reward
        srun += 1

    print(average_reward/p["stat_runs"])
    create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths")
    create_csv_file(reward_history, "Output_Data/", "Final_GlobalRewards.csv")
    create_csv_file(incursion_tracker, "Output_Data/", "HazardIncursions.csv")
    if p["vis_running"]:
        run_visualizer()


if __name__ == '__main__':
    # Test Performance of CBA
    counterfactuals = {}
    if p["c_type"] == "Best_Total":
        choices = range(p["n_skills"])
        n = p["n_rovers"]
        t_list = [choices] * n
        rover_skill_selections = np.zeros((p["n_skills"], p["n_skills"]))
        for srun in range(p["stat_runs"]):
            print(srun+1, "/", p["stat_runs"])
            c_list = (product(*t_list))
            counterfactuals["S{0}".format(srun)], r_skills = find_best_suggestions(srun, c_list)
            for rover_id in range(p["n_rovers"]):
                for c in range(p["n_skills"]):
                    rover_skill_selections[c][c] += r_skills["RV{0}".format(rover_id)][c][c]
        for rover_id in range(p["n_rovers"]):
            for c in range(p["n_skills"]):
                create_csv_file(rover_skill_selections[c], "Output_Data/", "Rover{0}_SkillSelections.csv".format(rover_id))
    elif p["c_type"] == "Best_Random":
        c_list = np.random.randint(0, p["n_skills"], (p["c_list_size"], p["n_rovers"]))
        rover_skill_selections = np.zeros((p["n_skills"], p["n_skills"]))
        for srun in range(p["stat_runs"]):
            print(srun+1, "/", p["stat_runs"])
            counterfactuals["S{0}".format(srun)], r_skills = find_best_suggestions(srun, c_list)
            for rover_id in range(p["n_rovers"]):
                for c in range(p["n_skills"]):
                    rover_skill_selections[c][c] += r_skills["RV{0}".format(rover_id)][c][c]
        for rover_id in range(p["n_rovers"]):
            for c in range(p["n_skills"]):
                create_csv_file(rover_skill_selections[c], "Output_Data/", "Rover{0}_SkillSelections.csv".format(rover_id))
    else:  # Custom
        rover_suggestions = [0, 0, 0]
        for srun in range(p["stat_runs"]):
            counterfactuals["S{0}".format(srun)] = rover_suggestions

    # Testing CBA using the selected set of counterfactual suggestions
    test_cba(counterfactuals)

