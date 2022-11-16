from ccea import Ccea
from rover_neural_network import NeuralNetwork
from RoverDomain_Core.rover_domain import RoverDomain
from ACG.supervisor import Supervisor
from ACG.supervisor_neural_network import SupervisorNetwork
from parameters import parameters as p
import numpy as np


def train_supervisor():
    """
    Train CBA rovers using a hand-crafted set of rover skills
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Supervisor Setup
    sup = Supervisor(n_agents=p["n_rovers"])
    sup_nn = SupervisorNetwork()

    # Create rover instances
    rovers = {}
    for rover_id in range(p["n_rovers"]):
        rovers["RV{0}".format(rover_id)] = NeuralNetwork(n_inp=p["s_inp"], n_hid=p["s_hid"], n_out=p["s_out"])
        # Import rover neural network weights from pickle
        weights = None
        rovers["RV{0}".format(rover_id)].get_weights(weights)  # CBA Network Gets Weights

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < p["stat_runs"]:
        print("Run: %i" % srun)

        # Create new supervisor EA pop

        policy_rewards = [[] for i in range(p["n_rovers"])]
        for gen in range(p["generations"]):
            # Each policy in EA is tested
            rover_rewards = np.zeros((p["n_rovers"], p["steps"]))  # Keep track of rover rewards at each t

            # Reset environment to initial conditions and select network weights
            rd.reset_world()
            for step_id in range(p["steps"]):
                # Rover scans environment and processes counterfactually shaped perceptions
                rover_actions = []
                for rv in rd.rovers:
                    rover_id = rd.rovers[rv].rover_id
                    rd.rovers[rv].scan_environment(rd.rovers, rd.pois)
                    sensor_data = rd.rovers[rv].observations  # Unaltered sensor readings

                    # Get counterfactual from supervisor

                    # Combine rover obervations with supervisor counterfactual

                    # Run rover neural network with counterfactual information

                # Calculate Rewards
                for rover_id in range(p["n_rovers"]):
                    reward = None
                    rover_rewards[rover_id, step_id] = reward

                # Update policy fitness

            # Choose parents and create new offspring population

        # Record trial data and supervisor network information

        srun += 1
