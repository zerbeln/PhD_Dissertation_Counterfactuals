from NeuralNetworks.neural_network import NeuralNetwork
from RoverDomainCore.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
import numpy as np
import sys
from parameters import parameters as p
from itertools import product
from CKI.custom_rover_skills import get_custom_action
from global_functions import *
from CKI.cki import get_counterfactual_state, calculate_poi_sectors


def find_best_counterfactuals(srun, c_list, config_id):
    """
    Find the best counterfactual states to use for the given environment
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()
    calculate_poi_sectors(rd.pois)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    networks = {}
    rover_skill_selections = {}
    for rover_id in range(p["n_rovers"]):
        networks[f'NN{rover_id}'] = NeuralNetwork(n_inp=p["cki_inp"], n_hid=p["cki_hid"], n_out=p["cki_out"])
        rover_skill_selections[f'RV{rover_id}'] = [[0 for i in range(p["n_skills"])] for j in range(p["n_skills"])]

    # Load Trained Suggestion Interpreter Weights
    for rover_id in range(p["n_rovers"]):
        s_weights = load_saved_policies(f'SelectionWeights{rover_id}', rover_id, srun)
        networks[f'NN{rover_id}'].get_weights(s_weights)

    best_rover_suggestion = None
    best_reward = None

    for c in c_list:
        c_state = [i for i in c]

        # Reset environment to initial conditions
        rd.reset_world(config_id)

        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        n_incursions = 0  # Number of times rovers violate a hazardous area
        for step_id in range(p["steps"]):
            rover_actions = []
            for rv in rd.rovers:
                rover_id = rd.rovers[rv].rover_id
                rd.rovers[rv].scan_environment(rd.rovers, rd.pois)
                sensor_data = rd.rovers[rv].observations  # Unaltered sensor readings

                # Select a skill using counterfactually shaped state information
                c_sensor_data = get_counterfactual_state(rd.pois, rd.rovers, rover_id, c_state[rover_id], sensor_data)
                cki_input = np.sum((c_sensor_data, sensor_data), axis=0)  # Shaped agent perception
                cki_outputs = networks[f'NN{rover_id}'].run_rover_nn(cki_input)  # CKI picks skill
                chosen_pol = int(np.argmax(cki_outputs))
                rover_skill_selections[f'RV{rover_id}'][c_state[rover_id]][chosen_pol] += 1

                # Determine action based on sensor inputs and suggestion
                action = get_custom_action(chosen_pol, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                rover_actions.append(action)

            # Environment takes in rover actions and returns next state and global reward
            step_rewards = rd.step(rover_actions)
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]
                if rd.pois[f'P{poi_id}'].hazardous:
                    for dist in rd.pois[f'P{poi_id}'].observer_distances:
                        if dist < p["observation_radius"]:
                            n_incursions += 1

        # Calculate episodic global reward
        g_reward = 0
        for p_reward in poi_rewards:
            g_reward += max(p_reward)
        g_reward -= (n_incursions * p["h_penalty"])
        if best_reward is None or g_reward > best_reward:
            best_reward = g_reward
            best_rover_suggestion = c_state

    create_csv_file(best_rover_suggestion, "Output_Data/", "BestRoverCounterfactuals.csv")

    return best_rover_suggestion, rover_skill_selections


def test_cki(counterfactuals, config_id):
    """
    Test CKI using the hand created rover policies
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()
    calculate_poi_sectors(rd.pois)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    networks = {}
    for rover_id in range(p["n_rovers"]):
        networks[f'NN{rover_id}'] = NeuralNetwork(n_inp=p["cki_inp"], n_hid=p["cki_hid"], n_out=p["cki_out"])

    average_reward = 0
    reward_history = []  # Keep track of team performance throughout training
    incursion_tracker = []  # Keep track of the number of hazard area violations each stat run
    final_rover_path = np.zeros((p["stat_runs"], p["n_rovers"], p["steps"], 3))

    srun = p["starting_srun"]
    while srun < p["stat_runs"]:  # Perform statistical runs
        sgst = counterfactuals[f'S{srun}']

        # Load Trained Suggestion Interpreter Weights
        for rover_id in range(p["n_rovers"]):
            s_weights = load_saved_policies(f'SelectionWeights{rover_id}', rover_id, srun)
            networks[f'NN{rover_id}'].get_weights(s_weights)

        # Reset environment to initial conditions
        rd.reset_world(config_id)
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
                cki_input = np.sum((c_sensor_data, sensor_data), axis=0)  # Shaped agent perception
                cki_outputs = networks[f'NN{rover_id}'].run_rover_nn(cki_input)  # CKI picks skill
                chosen_pol = int(np.argmax(cki_outputs))

                # Determine action based on sensor inputs and suggestion
                action = get_custom_action(chosen_pol, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                rover_actions.append(action)

            # Environment takes in rover actions and returns next state and global reward
            step_rewards = rd.step(rover_actions)
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]
                if rd.pois[f'P{poi_id}'].hazardous:
                    for dist in rd.pois[f'P{poi_id}'].observer_distances:
                        if dist < p["observation_radius"]:
                            n_incursions += 1

        # Calculate episodic global reward
        g_reward = 0
        for p_reward in poi_rewards:
            g_reward += max(p_reward)
        g_reward -= (n_incursions * p["h_penalty"])
        reward_history.append(g_reward)
        incursion_tracker.append(n_incursions)
        average_reward += g_reward
        srun += 1

    print(average_reward/p["stat_runs"])
    create_pickle_file(final_rover_path, "Output_Data/", f'Rover_Paths{config_id}')
    create_csv_file(reward_history, "Output_Data/", "TeamPerformance_CKI.csv")
    create_csv_file(incursion_tracker, "Output_Data/", "HazardIncursions.csv")
    if p["vis_running"]:
        run_visualizer(cf_id=config_id)


if __name__ == '__main__':
    config_id = int(sys.argv[1])
    print("Testing CKI on Configuration: ", config_id)

    # Test Performance of CKI
    counterfactuals = {}
    if p["c_type"] == 'Custom':
        rover_c_states = [0 for i in range(p["n_skills"])]
        for srun in range(p["stat_runs"]):
            counterfactuals[f'S{srun}'] = rover_c_states
    elif p["c_type"] == "Best_Total":
        choices = range(p["n_skills"])
        n = p["n_rovers"]
        t_list = [choices] * n
        rover_skill_selections = np.zeros((p["n_rovers"], p["n_skills"], p["n_skills"]))  # For heat map
        for srun in range(p["stat_runs"]):
            print(srun+1, "/", p["stat_runs"])
            c_list = (product(*t_list))
            counterfactuals[f'S{srun}'], r_skills = find_best_counterfactuals(srun, c_list, config_id)
            for rover_id in range(p["n_rovers"]):
                for c in range(p["n_skills"]):
                    for ci in range(p["n_skills"]):
                        rover_skill_selections[rover_id, c, ci] += r_skills[f'RV{rover_id}'][c][ci]
        for rover_id in range(p["n_rovers"]):
            for c in range(p["n_skills"]):
                create_csv_file(rover_skill_selections[rover_id, c], "Output_Data/", f'Rover{rover_id}_SkillSelections.csv')
    else:
        c_list = np.random.randint(0, p["n_skills"], (p["c_list_size"], p["n_rovers"]))
        rover_skill_selections = np.zeros((p["n_rovers"], p["n_skills"], p["n_skills"]))
        for srun in range(p["stat_runs"]):
            print(srun+1, "/", p["stat_runs"])
            counterfactuals[f'S{srun}'], r_skills = find_best_counterfactuals(srun, c_list, config_id)
            for rover_id in range(p["n_rovers"]):
                for c in range(p["n_skills"]):
                    for ci in range(p["n_skills"]):
                        rover_skill_selections[rover_id, c, ci] += r_skills[f'RV{rover_id}'][c][ci]
        for rover_id in range(p["n_rovers"]):
            for c in range(p["n_skills"]):
                create_csv_file(rover_skill_selections[rover_id, c], "Output_Data/", f'Rover{rover_id}_SkillSelections.csv')

    # Testing CKI using the selected set of counterfactual states
    test_cki(counterfactuals, config_id)

