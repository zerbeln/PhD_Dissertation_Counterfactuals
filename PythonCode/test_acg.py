from rover_neural_network import NeuralNetwork
from RoverDomain_Core.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
import numpy as np
from parameters import parameters as p
from ACG.supervisor import Supervisor
from ACG.supervisor_neural_network import SupervisorNetwork
from global_functions import *


def test_acg():
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    if p["active_hazards"]:
        for poi_id in p["hazardous_poi"]:
            rd.pois["P{0}".format(poi_id)].hazardous = True

    # Supervisor Setup
    sup = Supervisor()
    sup_nn = SupervisorNetwork(n_inp=p["acg_inp"], n_hid=p["acg_hid"], n_out=p["acg_out"], n_agents=p["n_rovers"])

    rover_networks = {}
    for rover_id in range(p["n_rovers"]):
        rover_networks["NN{0}".format(rover_id)] = NeuralNetwork(n_inp=p["n_inp"], n_hid=p["n_hid"], n_out=p["n_out"])

    average_reward = 0
    reward_history = []  # Keep track of team performance throughout training
    incursion_tracker = []  # Keep track of the number of hazard area violations each stat run
    final_rover_path = np.zeros((p["stat_runs"], p["n_rovers"], p["steps"], 3))

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        # Load Trained Networks for Rovers and Supervisor
        s_weights = load_saved_policies('SupervisorWeights', p["n_rovers"]+1, srun)
        sup_nn.get_weights(s_weights)
        for rover_id in range(p["n_rovers"]):
            weights = load_saved_policies('RoverWeights{0}'.format(rover_id), rover_id, srun)
            rover_networks["NN{0}".format(rover_id)].get_weights(weights)

        # Reset environment to initial conditions
        rd.reset_world()
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
                action = rover_networks["NN{0}".format(rover_id)].run_rover_nn(rover_input)  # CBA picks skill
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
        g_reward -= (n_incursions * 10)
        reward_history.append(g_reward)
        incursion_tracker.append(n_incursions)
        average_reward += g_reward
        srun += 1

    print(average_reward / p["stat_runs"])
    create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths")
    create_csv_file(reward_history, "Output_Data/", "TeamPerformance_ACG.csv")
    create_csv_file(incursion_tracker, "Output_Data/", "HazardIncursions.csv")
    if p["vis_running"]:
        run_visualizer()


if __name__ == '__main__':
    # Test Performance of Supervisor in ACG
    test_acg()
