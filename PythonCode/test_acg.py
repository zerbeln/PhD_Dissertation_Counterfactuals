from rover_neural_network import NeuralNetwork
from RoverDomain_Core.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
import numpy as np
import sys
from parameters import parameters as p
from ACG.supervisor import Supervisor
from ACG.supervisor_neural_network import SupervisorNetwork
from global_functions import *
from CBA.custom_rover_skills import get_custom_action


def test_acg_hazards(config_id):
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Supervisor Setup
    sup = Supervisor()
    sup_nn = SupervisorNetwork(n_inp=p["acg_inp"], n_hid=p["acg_hid"], n_out=p["acg_out"], n_agents=p["n_rovers"])

    rover_nns = {}
    for rover_id in range(p["n_rovers"]):
        rover_nns["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["cba_inp"], n_hid=p["cba_hid"], n_out=p["cba_out"])

    average_reward = 0
    reward_history = []  # Keep track of team performance throughout training
    incursion_tracker = []  # Keep track of the number of hazard area violations each stat run
    final_rover_path = np.zeros((p["stat_runs"], p["n_rovers"], p["steps"], 3))

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        # Load Trained Networks for Rovers and Supervisor
        s_weights = load_saved_policies('SupervisorWeights', p["n_rovers"], srun)
        sup_nn.get_weights(s_weights)
        for rover_id in range(p["n_rovers"]):
            weights = load_saved_policies('RoverWeights{0}'.format(rover_id), rover_id, srun)
            rover_nns["NN{0}".format(rover_id)].get_weights(weights)

        # Reset environment to initial conditions
        rd.reset_world(config_id)
        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        n_incursions = 0
        for step_id in range(p["steps"]):
            # Supervisor observes environment and creates counterfactuals
            sup.scan_environment(rd.rovers, rd.pois)
            counterfactuals = sup_nn.run_supervisor_nn(sup.observations)

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

                # Rover acts based on perception + supervisor counterfactual
                c_data = counterfactuals["RV{0}".format(rover_id)]  # Counterfactual from supervisor
                rover_input = np.sum((sensor_data, c_data), axis=0)
                nn_output = rover_nns["NN{0}".format(rover_id)].run_rover_nn(rover_input)  # CBA picks skill
                chosen_pol = int(np.argmax(nn_output))
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
        g_reward -= (n_incursions * p["h_penalty"])
        reward_history.append(g_reward)
        incursion_tracker.append(n_incursions)
        average_reward += g_reward
        srun += 1

    print(average_reward / p["stat_runs"])
    create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths{0}".format(config_id))
    create_csv_file(reward_history, "Output_Data/", "TeamPerformance_ACG.csv")
    create_csv_file(incursion_tracker, "Output_Data/", "HazardIncursions.csv")
    if p["vis_running"]:
        run_visualizer(cf_id=config_id)


def test_acg_rover_loss(config_id, lost_rovers):
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Supervisor Setup
    sup = Supervisor()
    sup_nn = SupervisorNetwork(n_inp=p["acg_inp"], n_hid=p["acg_hid"], n_out=p["acg_out"], n_agents=p["n_rovers"])

    rover_nns = {}
    for rover_id in range(p["n_rovers"]):
        rover_nns["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["cba_inp"], n_hid=p["cba_hid"], n_out=p["cba_out"])

    average_reward = 0
    reward_history = []  # Keep track of team performance throughout training
    final_rover_path = np.zeros((p["stat_runs"], p["n_rovers"], p["steps"], 3))

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        # Load Trained Networks for Rovers and Supervisor
        s_weights = load_saved_policies('SupervisorWeights', p["n_rovers"], srun)
        sup_nn.get_weights(s_weights)
        for rover_id in range(p["n_rovers"]):
            weights = load_saved_policies('RoverWeights{0}'.format(rover_id), rover_id, srun)
            rover_nns["NN{0}".format(rover_id)].get_weights(weights)

        # Reset environment to initial conditions
        rd.reset_world(config_id)
        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        for step_id in range(p["steps"]):
            # Supervisor observes environment and creates counterfactuals
            sup.scan_environment(rd.rovers, rd.pois)
            counterfactuals = sup_nn.run_supervisor_nn(sup.observations)

            rover_actions = []
            for rv in rd.rovers:
                # Record rover paths
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 0] = rd.rovers[rv].loc[0]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 1] = rd.rovers[rv].loc[1]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 2] = rd.rovers[rv].loc[2]

                # Rovers observe environment
                if rd.rovers[rv].rover_id not in lost_rovers:
                    rover_id = rd.rovers[rv].rover_id
                    rd.rovers[rv].scan_environment(rd.rovers, rd.pois)
                    sensor_data = rd.rovers[rv].observations  # Unaltered sensor readings

                    # Rover acts based on perception + supervisor counterfactual
                    c_data = counterfactuals["RV{0}".format(rover_id)]  # Counterfactual from supervisor
                    rover_input = np.sum((sensor_data, c_data), axis=0)
                    nn_output = rover_nns["NN{0}".format(rover_id)].run_rover_nn(rover_input)  # CBA picks skill
                    chosen_pol = int(np.argmax(nn_output))
                    action = get_custom_action(chosen_pol, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                else:
                    action = [0.5, 0.5]  # Non-functional rovers do not move (0.5 cancels out with rover movement)

                rover_actions.append(action)

            # Environment takes in rover actions and returns next state and global reward
            step_rewards = rd.step(rover_actions)
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]

        # Calculate episodic global reward
        g_reward = 0
        for p_reward in poi_rewards:
            g_reward += max(p_reward)
        reward_history.append(g_reward)
        average_reward += g_reward
        srun += 1

    print(average_reward / p["stat_runs"])
    create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths{0}".format(config_id))
    create_csv_file(reward_history, "Output_Data/", "TeamPerformance_ACG.csv")
    if p["vis_running"]:
        run_visualizer(cf_id=config_id)


def test_standard_pol_rov_loss(config_id, lost_rovers):
    """
    Test rover policy trained using Global, Difference, or D++ rewards.
    """
    # World Setup
    rd = RoverDomain()  # Create instance of the rover domain
    rd.load_world()

    networks = {}
    for rover_id in range(p["n_rovers"]):
        networks["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["cba_inp"], n_hid=p["cba_hid"], n_out=p["cba_out"])

    # Data tracking
    reward_history = []  # Keep track of team performance throughout training
    average_reward = 0  # Track average reward across runs
    final_rover_path = np.zeros((p["stat_runs"], p["n_rovers"], p["steps"], 3))  # Track rover trajectories

    # Run tests
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        # Load Trained Rover Networks
        for rv in rd.rovers:
            rover_id = rd.rovers[rv].rover_id
            weights = load_saved_policies('RoverWeights{0}'.format(rover_id), rover_id, srun)
            networks["NN{0}".format(rd.rovers[rv].rover_id)].get_weights(weights)

        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        rd.reset_world(config_id)
        for step_id in range(p["steps"]):
            # Get rover actions from neural network
            rover_actions = []
            for rv in rd.rovers:
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 0] = rd.rovers[rv].loc[0]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 1] = rd.rovers[rv].loc[1]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 2] = rd.rovers[rv].loc[2]

                if rd.rovers[rv].rover_id not in lost_rovers:
                    # Get actions from rover neural networks
                    rover_id = rd.rovers[rv].rover_id
                    nn_output = networks["NN{0}".format(rover_id)].run_rover_nn(rd.rovers[rv].observations)
                    chosen_pol = int(np.argmax(nn_output))
                    action = get_custom_action(chosen_pol, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                    rover_actions.append(action)
                else:
                    rover_actions.append([0.5, 0.5])

            # Environment takes in rover actions and returns next state and global reward
            step_rewards = rd.step(rover_actions)
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]

        # Calculate episodic global reward
        g_reward = 0
        for p_reward in poi_rewards:
            g_reward += max(p_reward)
        reward_history.append(g_reward)
        average_reward += g_reward
        srun += 1

    print(average_reward/p["stat_runs"])
    create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths{0}".format(config_id))
    create_csv_file(reward_history, "Output_Data/", "Final_GlobalRewards.csv")
    if p["vis_running"]:
        run_visualizer(cf_id=config_id)


if __name__ == '__main__':
    # Test Performance of Supervisor in ACG
    cf_id = int(sys.argv[1])
    # cf_id = 0
    print("Testing ACG on Configuration: ", cf_id)

    if p["sup_train"] == "Hazards":
        test_acg_hazards(cf_id)
    elif p["sup_train"] == "Rover_Loss":
        lost_rovers = [1]  # Rovers that will be non-functional
        print("Testing Standard Policy")
        test_standard_pol_rov_loss(cf_id, lost_rovers)
        print("Testing ACG Policy")
        test_acg_rover_loss(cf_id, lost_rovers)
    else:
        print("INCORRECT SUPERVISOR TRAINING TYPE")
