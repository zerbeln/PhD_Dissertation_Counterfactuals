from Python_Code.ccea import Ccea
from Python_Code.suggestion_network import SuggestionNetwork
from Python_Code.rover_domain import RoverDomain
from Python_Code.agent import Rover
from Visualizer.visualizer import run_visualizer
import pickle
import csv
import os
import sys
import math
import numpy as np
from parameters import parameters as p


# Holds one suggestion along with the time it should start
class Suggestion:
    def __init__(self, value, timestamp=0):
        '''
        :param value: what policy to suggest
        :param timestamp: after what step to pick this policy
        :return:
        '''

        # These should not be modified after init, hence the getters
        self._value = value
        self._timestamp = timestamp

    def get_value(self):
        return self._value

    def get_timestamp(self):
        return self._timestamp


# Wraps a list of suggestions to improve readability
# Warning: does not enforce sorted timestamps-- add suggestions in order
class Suggestion_Queue:
    def __init__(self, prebuilt=None):
        if prebuilt is None:
            self.queue = [Suggestion(0)]
        else:
            assert isinstance(prebuilt, list)
            assert all(isinstance(x, Suggestion) for x in prebuilt)
            self.queue = prebuilt

    def add_suggestion(self, sgst):
        assert isinstance(sgst, (list, Suggestion))
        if isinstance(sgst, list):
            assert all(isinstance(x, Suggestion) for x in sgst)
            self.queue += sgst
        else:
            self.queue.append(sgst)

    # Get top suggestion, remove it if the next one should start
    def get_suggestion(self, timestamp):
        if len(self.queue) == 1:
            return self.queue[0].get_value()

        if self.queue[1].get_timestamp() <= timestamp:
            self.queue.pop(0)
        return self.queue[0].get_value()


def save_reward_history(reward_history, file_name):
    """
    Save the reward history for the agents throughout the learning process (reward from best policy team each gen)
    """

    dir_name = 'Output_Data/'  # Intended directory for output files
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, file_name)
    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


def save_rover_path(rover_path, file_name):  # Save path rovers take using best policy found
    """
    Records the path each rover takes using best policy from CCEA (used by visualizer)
    :param rover_path:  trajectory tracker
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    rpath_name = os.path.join(dir_name, file_name)
    rover_file = open(rpath_name, 'wb')
    pickle.dump(rover_path, rover_file)
    rover_file.close()


def load_saved_policies(file_name, rover_id, srun):
    """
    Load saved Neural Network policies from pickle file
    """

    dir_name = 'Policy_Bank/Rover{0}/SRUN{1}'.format(rover_id, srun)
    fpath_name = os.path.join(dir_name, file_name)
    weight_file = open(fpath_name, 'rb')
    weights = pickle.load(weight_file)
    weight_file.close()

    return weights


def create_policy_playbook(playbook_type, srun, n_inp, n_out, n_hid):
    """
    Choose which playbook of policies to load for the rovers
    """
    policy_bank = {}

    if playbook_type == "Four_Quadrants":
        for rover_id in range(p["n_rovers"]):
            w0 = load_saved_policies("TowardQuadrant0", rover_id, srun)
            w1 = load_saved_policies("TowardQuadrant1", rover_id, srun)
            w2 = load_saved_policies("TowardQuadrant2", rover_id, srun)
            w3 = load_saved_policies("TowardQuadrant3", rover_id, srun)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 0)] = SuggestionNetwork(n_inp, n_out, n_hid)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 0)].get_weights(w0)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 1)] = SuggestionNetwork(n_inp, n_out, n_hid)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 1)].get_weights(w1)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 2)] = SuggestionNetwork(n_inp, n_out, n_hid)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 2)].get_weights(w2)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 3)] = SuggestionNetwork(n_inp, n_out, n_hid)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 3)].get_weights(w3)
    elif playbook_type == "Two_POI":
        for rover_id in range(p["n_rovers"]):
            w0 = load_saved_policies("TowardPOI0", rover_id, srun)
            w1 = load_saved_policies("TowardPOI1", rover_id, srun)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 0)] = SuggestionNetwork(n_inp, n_out, n_hid)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 0)].get_weights(w0)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 1)] = SuggestionNetwork(n_inp, n_out, n_hid)
            policy_bank["Rover{0}Policy{1}".format(rover_id, 1)].get_weights(w1)

    return policy_bank


def get_angle_dist(x, y, tx, ty):
    """
    Computes angles and distance between two predators relative to (1,0) vector (x-axis)
    :param tx: X-Position of sensor target
    :param ty: Y-Position of sensor target
    :param x: X-Position of scanning rover
    :param y: Y-Position of scanning rover
    :return: angle, dist
    """

    vx = x - tx
    vy = y - ty

    angle = math.atan2(vy, vx)*(180.0/math.pi)

    while angle < 0.0:
        angle += 360.0
    while angle > 360.0:
        angle -= 360.0
    if math.isnan(angle):
        angle = 0.0

    dist = (vx**2) + (vy**2)

    # Clip distance to not overwhelm activation function in NN
    if dist < p["dmax"]:
        dist = p["dmax"]

    return angle, dist


def construct_counterfactual_state(poi_info, rover_info, rover_id, suggestion):
    """
    Create a counteractual state input to represent agent suggestions
    """
    rover_pos = rover_info["Rover{0}".format(rover_id)].pos
    cfact_poi, poi_quadrants = create_counterfactual_poi_state(poi_info, rover_pos, suggestion)
    cfact_rover = create_counterfactual_rover_state(rover_info, rover_pos, rover_id, poi_quadrants, suggestion)

    counterfactual_state = np.zeros(8)
    for i in range(4):
        counterfactual_state[i] = cfact_poi[i]
        counterfactual_state[4 + i] = cfact_rover[i]

    return counterfactual_state


def create_counterfactual_poi_state(poi_info, rover_pos, suggestion):
    """
    Construct a counterfactual state input for POI detections
    :return: Portion of the counterfactual state constructed from POI scanner
    """
    poi_state = np.zeros(int(360.0 / p["angle_res"]))
    temp_poi_dist_list = [[] for _ in range(int(360.0 / p["angle_res"]))]
    poi_quadrants = np.zeros(p["n_poi"], int)

    # Log POI distances into brackets
    n_poi = len(poi_info)
    for poi_id in range(n_poi):
        angle, dist = get_angle_dist(rover_pos[0], rover_pos[1], poi_info[poi_id, 0], poi_info[poi_id, 1])

        bracket = int(angle / p["angle_res"])
        if bracket > 3:
            bracket -= 4
        poi_quadrants[poi_id] = bracket
        if poi_info[poi_id, 3] == suggestion:
            temp_poi_dist_list[bracket].append(10*poi_info[poi_id, 2] / dist)

    # Encode POI information into the state vector
    for bracket in range(int(360 / p["angle_res"])):
        num_poi_bracket = len(temp_poi_dist_list[bracket])  # Number of POIs in bracket
        if num_poi_bracket > 0:
            if p["sensor_model"] == 'density':
                poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi_bracket  # Density Sensor
            elif p["sensor_model"] == 'summed':
                poi_state[bracket] = sum(temp_poi_dist_list[bracket])  # Summed Distance Sensor
            else:
                sys.exit('Incorrect sensor model')
        else:
            poi_state[bracket] = -1.0

    return poi_state, poi_quadrants


def create_counterfactual_rover_state(rover_info, rover_pos, self_id, poi_quadrants, suggestion):
    """
    Construct a counterfactual state input for rover detections
    :return: Portion of the counterfactual state vector created from rover scanner
    """
    center_x = p["x_dim"]
    center_y = p["y_dim"]
    rover_state = np.zeros(int(360.0 / p["angle_res"]))
    temp_rover_dist_list = [[] for _ in range(int(360.0 / p["angle_res"]))]

    # Log rover distances into brackets
    for rover_id in range(p["n_rovers"]):
        if self_id != rover_id:  # Ignore self
            rov_x = rover_info["Rover{0}".format(rover_id)].pos[0]
            rov_y = rover_info["Rover{0}".format(rover_id)].pos[1]

            angle, dist = get_angle_dist(rover_pos[0], rover_pos[1], rov_x, rov_y)
            bracket = int(angle / p["angle_res"])
            if bracket > 3:
                bracket -= 4

            w_angle, w_dist = get_angle_dist(center_x, center_y, rov_x, rov_y)
            world_bracket = int(w_angle/p["angle_res"])
            if world_bracket > 3:
                world_bracket -= 4

            if suggestion == world_bracket:
                temp_rover_dist_list[bracket].append(10 / dist)

    # Encode Rover information into the state vector
    for bracket in range(int(360 / p["angle_res"])):
        num_rovers_bracket = len(temp_rover_dist_list[bracket])  # Number of rovers in bracket
        if num_rovers_bracket > 0:
            if p["sensor_model"] == 'density':
                rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_rovers_bracket  # Density Sensor
            elif p["sensor_model"] == 'summed':
                rover_state[bracket] = sum(temp_rover_dist_list[bracket])  # Summed Distance Sensor
            else:
                sys.exit('Incorrect sensor model')
        else:
            rover_state[bracket] = -1.0

    return rover_state


def test_suggestions_policy_bank(pbank_type, sgst):
    """
    Test suggestions using the pre-trained policy bank
    """
    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    domain_type = p["domain_type"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # Suggestion Parameters
    s_inp = p["s_inputs"]
    s_hid = p["s_hidden"]
    s_out = p["s_outputs"]

    rd = RoverDomain()  # Create instance of the rover domain

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["SN{0}".format(rover_id)] = SuggestionNetwork(s_inp, s_out, s_hid)

    average_reward = 0
    reward_history = []  # Keep track of team performance throughout training
    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs

        # Load Trained Suggestion Interpreter Weights
        for rover_id in range(n_rovers):
            s_weights = load_saved_policies('SelectionWeights{0}'.format(rover_id), rover_id, srun)
            rovers["SN{0}".format(rover_id)].get_weights(s_weights)

        # Load World Configuration
        rd.load_world(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
            final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
            final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]

        # Load Pre-Trained Policies
        policy_bank = create_policy_playbook(pbank_type, srun, n_inp, n_out, n_hid)

        for rover_id in range(n_rovers):  # Initial rover scan of environment
#            this_rover_sgst = sgst[rover_id][0].get_value()
            this_rover_sgst = sgst[rover_id].get_suggestion(0)

            suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, this_rover_sgst)
            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
            sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
            sug_input = np.concatenate((suggestion, sensor_data), axis=0)
            rovers["SN{0}".format(rover_id)].get_inputs(sug_input)
            sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
            pol_id = np.argmax(sug_outputs)
            rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
            rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions

        g_rewards = np.zeros(rover_steps)
        for step_id in range(rover_steps):
            # Rover takes an action in the world
            for rover_id in range(n_rovers):
                rovers["Rover{0}".format(rover_id)].suggestion_step(rd.world_x, rd.world_y)
                final_rover_path[srun, rover_id, step_id + 1, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                final_rover_path[srun, rover_id, step_id + 1, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                final_rover_path[srun, rover_id, step_id + 1, 2] = rovers["Rover{0}".format(rover_id)].pos[2]

            # Rover scans environment and processes suggestions
            for rover_id in range(n_rovers):
                # Get current suggestion, remove from list if expired
#                this_rover_sgst = sgst[rover_id][0].get_value()
#                if len(sgst[rover_id]) > 1 and sgst[rover_id][1].get_timestamp() <= step_id:
#                    sgst[rover_id].pop(0)
                this_rover_sgst = sgst[rover_id].get_suggestion(step_id)

                if rover_id == 0:
                    print(step_id, this_rover_sgst)

                suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, this_rover_sgst)
                rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                rovers["SN{0}".format(rover_id)].get_inputs(np.concatenate((suggestion, sensor_data), axis=0))
                sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                pol_id = np.argmax(sug_outputs)
                rv_actions = policy_bank["Rover{0}Policy{1}".format(rover_id, pol_id)].run_network(sensor_data)
                rovers["Rover{0}".format(rover_id)].rover_actions = rv_actions

            # Calculate Global Reward
            if domain_type == "Loose":
                g_rewards[step_id] = rd.calc_global_loose()
            else:
                g_rewards[step_id] = rd.calc_global_tight()

        reward_history.append(sum(g_rewards))
        average_reward += sum(g_rewards)

        save_rover_path(final_rover_path, "Rover_Paths")
        # print(reward_history)
    average_reward /= stat_runs
    print(average_reward)
    save_reward_history(reward_history, "Final_GlobalRewards.csv")
    run_visualizer()


if __name__ == '__main__':

    # Create list of suggestions for rovers to use during training
    rover_suggestions = {}
    if p["policy_bank_type"] == "Two_POI":
        for rover_id in range(p["n_rovers"]):
            rover_suggestions[rover_id] = Suggestion_Queue([Suggestion(0), Suggestion(1,30)])
    elif p["policy_bank_type"] == "Four_Quadrants":
        for rover_id in range(p["n_rovers"]):
            sugg = 0
            rover_suggestions.append(sugg)
    test_suggestions_policy_bank(p["policy_bank_type"], rover_suggestions)

