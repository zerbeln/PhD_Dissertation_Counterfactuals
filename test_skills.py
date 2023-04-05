from CKI.cki import calculate_poi_sectors
from RoverDomainCore.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
from NeuralNetworks.neural_network import NeuralNetwork
import numpy as np
from parameters import parameters as p
from global_functions import create_csv_file, create_pickle_file, load_saved_policies
from CKI.custom_rover_skills import get_custom_action


def test_custom_skills(skill_id, config_id):
    """
    Test suggestions using the pre-trained policy bank
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()
    calculate_poi_sectors(rd.pois)

    final_rover_path = np.zeros((p["stat_runs"], p["n_rovers"], p["steps"], 3))
    srun = p["starting_srun"]
    skill_performance = []  # Keep track of team performance throughout training
    while srun < p["stat_runs"]:
        # Reset rover and record initial position
        rd.reset_world(config_id)
        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        for step_id in range(p["steps"]):
            # Rover takes an action in the world
            rover_actions = []
            for rv in rd.rovers:
                action = get_custom_action(skill_id, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                rover_actions.append(action)
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 0] = rd.rovers[rv].loc[0]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 1] = rd.rovers[rv].loc[1]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 2] = rd.rovers[rv].loc[2]

            # Environment takes in rover actions and returns next state and global reward
            step_rewards = rd.step(rover_actions)
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]

        # Calculate episodic global reward
        g_reward = 0
        for p_reward in poi_rewards:
            g_reward += max(p_reward)
        skill_performance.append(g_reward)
        srun += 1

    create_pickle_file(final_rover_path, "Output_Data/", f'Rover_Paths{config_id}')
    create_csv_file(skill_performance, "Output_Data/", f'Skill{skill_id}_Performance.csv')
    if p["vis_running"]:
        run_visualizer(cf_id=config_id)


def test_trained_policy(config_id):
    """
    Test rover policy trained using Global, Difference, or D++ rewards.
    """
    # World Setup
    rd = RoverDomain()  # Create instance of the rover domain
    rd.load_world()

    networks = {}
    for rover_id in range(p["n_rovers"]):
        networks[f'NN{rover_id}'] = NeuralNetwork(n_inp=p["cki_inp"], n_hid=p["cki_hid"], n_out=p["cki_out"])

    # Data tracking
    reward_history = []  # Keep track of team performance throughout training
    incursion_tracker = []  # Keep track of the number of hazard area violations each stat run
    average_reward = 0  # Track average reward across runs
    final_rover_path = np.zeros((p["stat_runs"], p["n_rovers"], p["steps"], 3))  # Track rover trajectories

    # Run tests
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        rover_poi_tracker = np.zeros((p["n_rovers"], p["n_poi"] + 1))  # Track number of times rovers visit POI or none
        # Load Trained Rover Networks
        for rv in rd.rovers:
            rover_id = rd.rovers[rv].rover_id
            weights = load_saved_policies(f'RoverWeights{rover_id}', rover_id, srun)
            networks[f'NN{rover_id}'].get_weights(weights)

        n_incursions = 0
        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        rd.reset_world(config_id)
        for step_id in range(p["steps"]):
            # Get rover actions from neural network
            rover_actions = []
            for rv in rd.rovers:
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 0] = rd.rovers[rv].loc[0]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 1] = rd.rovers[rv].loc[1]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 2] = rd.rovers[rv].loc[2]

                # Get actions from rover neural networks
                nn_output = networks[f'NN{rd.rovers[rv].rover_id}'].run_rover_nn(rd.rovers[rv].observations)
                chosen_pol = int(np.argmax(nn_output))
                action = get_custom_action(chosen_pol, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                rover_actions.append(action)

            # Environment takes in rover actions and returns next state and global reward
            step_rewards = rd.step(rover_actions)
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]
                rov_id = 0
                for dist in rd.pois[f'P{poi_id}'].observer_distances:
                    if dist < p["observation_radius"]:
                        rover_poi_tracker[rov_id, poi_id] += 1
                        if rd.pois[f'P{poi_id}'].hazardous:
                            n_incursions += 1
                    rov_id += 1

        for rov_id in range(p["n_rovers"]):
            none_visited = True
            for poi_id in range(p["n_poi"]):
                if rover_poi_tracker[rov_id, poi_id] > 0:
                    rover_poi_tracker[rov_id, poi_id] = 1
                    none_visited = False
            if none_visited:
                rover_poi_tracker[rov_id, p["n_poi"]] += 1

        # Calculate episodic global reward
        g_reward = 0
        for p_reward in poi_rewards:
            g_reward += max(p_reward)
        g_reward -= (n_incursions * p["h_penalty"])
        reward_history.append(g_reward)
        incursion_tracker.append(n_incursions)
        average_reward += g_reward
        for rover_id in range(p["n_rovers"]):
            create_csv_file(rover_poi_tracker[rover_id], "Output_Data/", f'Rover{rover_id}POIVisits.csv')
        srun += 1

    print(average_reward/p["stat_runs"])
    create_pickle_file(final_rover_path, "Output_Data/", f'Rover_Paths{config_id}')
    create_csv_file(reward_history, "Output_Data/", "Final_GlobalRewards.csv")
    create_csv_file(incursion_tracker, "Output_Data/", "HazardIncursions.csv")
    if p["vis_running"]:
        run_visualizer(cf_id=config_id)


if __name__ == '__main__':
    # Test Performance of Skills in Agent Skill Set
    cf_id = 0
    skill_id = 5
    print("Testing Trained Skills on Configuration: ", cf_id)
    test_custom_skills(skill_id, cf_id)
