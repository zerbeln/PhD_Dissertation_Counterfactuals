import numpy as np
import copy
from RewardFunctions.reward_functions import calc_difference
from parameters import parameters as p


# S-Difference REWARD -------------------------------------------------------------------------------------------------
def calc_sd_reward(observer_distances, poi, global_reward, sgst):
    """
    Calculate each rover's difference reward with counterfactual suggestions at the current time step
    :param observer_distances: Each rover's distance to each POI
    :param poi: np array with X-Y coordinates and value for each POI
    :param global_reward: Reward given to the team from the world
    :return: difference_rewards (np array of size (n_rovers))
    """
    obs_rad = p["observation_radius"]
    cpl = p["coupling"]
    n_poi = p["n_poi"]
    n_rovers = p["n_rovers"]

    difference_rewards = np.zeros(n_rovers)
    for agent_id in range(n_rovers):  # For each rover
        counterfactual_global_reward = 0.0
        s_id = sgst[agent_id]
        for poi_id in range(n_poi):  # For each POI
            observer_count = 0
            rover_distances = observer_distances[poi_id].copy()
            if poi[poi_id, 3] == s_id:
                rover_distances[agent_id] = 1000.00
            rover_distances = np.sort(rover_distances)

            for i in range(cpl):
                dist = rover_distances[i]
                if dist < obs_rad:
                    observer_count += 1

            # Compute reward if coupling is satisfied
            if observer_count >= cpl:
                summed_observer_distances = sum(rover_distances[0:cpl])
                counterfactual_global_reward += poi[poi_id, 2] / (summed_observer_distances/cpl)

        difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards


# S-D++ REWARD -------------------------------------------------------------------------------------------------------
def calc_sdpp(pois, poi_rewards, sgst):
    """
    Calculate D++ rewards with counterfactual suggestions for each rover at the current time step
    :return: dpp_rewards (np array of size (n_rovers))
    """
    n_poi = p["n_poi"]
    n_rovers = p["n_rovers"]
    obs_rad = p["observation_radius"]
    global_reward = sum(poi_rewards)  # Current global reward at current time step
    d_rewards = calc_difference(pois, poi_rewards)
    rewards = np.zeros((n_rovers, n_poi))
    sdpp_rewards = np.zeros(n_rovers)

    # Calculate S-D++ Reward with (TotalAgents - 1) Counterfactuals
    for agent_id in range(n_rovers):
        s_id = sgst[agent_id]
        for pk in pois:
            observer_count = 0
            counterfactual_global_reward = 0.0
            n_counters = pois[pk].coupling - 1
            rover_distances = copy.deepcopy(pois[pk].observer_distances)
            counterfactual_rovers = np.ones(int(n_counters))
            if pois[pk].poi_id == s_id:
                counterfactual_rovers *= pois[pk].observer_distances[agent_id]
            else:
                counterfactual_rovers *= 1000.00
            rover_distances = np.append(rover_distances, counterfactual_rovers)
            rover_distances = np.sort(rover_distances)

            for i in range(int(pois[pk].coupling)):
                if rover_distances[i] < obs_rad:
                    observer_count += 1

            # Update POI observers
            if observer_count >= pois[pk].coupling:
                summed_observer_distances = sum(rover_distances[0:int(pois[pk].coupling)])
                counterfactual_global_reward += pois[pk].value/(summed_observer_distances / pois[pk].coupling)

            rewards[agent_id, pois[pk].poi_id] = (counterfactual_global_reward - global_reward)/n_counters

    for agent_id in range(n_rovers):
        if sum(rewards[agent_id, :]) > d_rewards[agent_id]:
            s_id = sgst[agent_id]
            for pk in pois:
                n_counters = 1
                while n_counters < pois[pk].coupling - 1:
                    observer_count = 0
                    counterfactual_global_reward = 0.0
                    rover_distances = copy.deepcopy(pois[pk].observer_distances)
                    counterfactual_rovers = np.ones(int(n_counters))
                    if pois[pk].poi_id == s_id:
                        counterfactual_rovers *= pois[pk].observer_distances[agent_id]
                    else:
                        counterfactual_rovers *= 1000.00
                    rover_distances = np.append(rover_distances, counterfactual_rovers)
                    rover_distances = np.sort(rover_distances)

                    for i in range(int(pois[pk].coupling)):
                        if rover_distances[i] < obs_rad:
                            observer_count += 1

                    # Update POI observers
                    if observer_count >= pois[pk].coupling:
                        summed_observer_distances = sum(rover_distances[0:int(pois[pk].coupling)])
                        counterfactual_global_reward += pois[pk].value/(summed_observer_distances/pois[pk].coupling)

                    # Calculate S-D++ reward with n counterfactuals added
                    temp_dpp = (counterfactual_global_reward - global_reward)/n_counters
                    if temp_dpp > rewards[agent_id, pois[pk].poi_id]:
                        rewards[agent_id, pois[pk].poi_id] = temp_dpp
                        n_counters = pois[pk].coupling + 1  # Stop iterrating
                    else:
                        n_counters += 1
            sdpp_rewards[agent_id] = sum(rewards[agent_id, :])
        else:
            sdpp_rewards[agent_id] = d_rewards[agent_id]  # Returns difference reward

    return sdpp_rewards


def sdpp_and_sd(observer_distances, poi, global_reward, sgst):
    """
    Calculate D++ rewards and difference rewards for each rover using counterfactual suggestions
    :param observer_distances: Each rover's distance to each POI
    :param poi: np array with X-Y coordinates and value for each POI
    :param global_reward: Reward given to the team from the world
    :return: dpp_rewards (np array of size (n_rovers))
    """

    n_poi = p["n_poi"]
    n_rovers = p["n_rovers"]
    cpl = p["coupling"]
    obs_rad = p["observation_radius"]
    d_rewards = calc_sd_reward(observer_distances, poi, global_reward, sgst)
    dpp_rewards = np.zeros(n_rovers)

    # Calculate S-D++ Reward with (TotalAgents - 1) Counterfactuals
    n_counters = cpl - 1
    for agent_id in range(n_rovers):
        counterfactual_global_reward = 0.0
        s_id = sgst[agent_id]
        for poi_id in range(n_poi):
            observer_count = 0
            rover_distances = observer_distances[poi_id].copy()
            counterfactual_rovers = np.ones(n_counters)
            if poi[poi_id, 3] == s_id:
                counterfactual_rovers *= observer_distances[poi_id, agent_id]
            else:
                counterfactual_rovers *= 1000.00
            rover_distances = np.append(rover_distances, counterfactual_rovers)
            rover_distances = np.sort(rover_distances)

            for i in range(cpl):
                if rover_distances[i] < obs_rad:
                    observer_count += 1

            # Update POI observers
            if observer_count >= cpl:
                summed_observer_distances = sum(rover_distances[0:cpl])
                counterfactual_global_reward += poi[poi_id, 2] / (summed_observer_distances / cpl)

        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / n_counters

    for agent_id in range(n_rovers):
        if dpp_rewards[agent_id] > d_rewards[agent_id]:
            dpp_rewards[agent_id] = 0.0
            s_id = sgst[agent_id]
            for c in range(n_counters):
                n_counters = c + 1
                counterfactual_global_reward = 0.0
                for poi_id in range(n_poi):
                    observer_count = 0
                    rover_distances = observer_distances[poi_id].copy()
                    counterfactual_rovers = np.ones(n_counters)
                    if poi[poi_id, 3] == s_id:
                        counterfactual_rovers *= observer_distances[poi_id, agent_id]
                    else:
                        counterfactual_rovers *= 1000.00
                    rover_distances = np.append(rover_distances, counterfactual_rovers)
                    rover_distances = np.sort(rover_distances)

                    for i in range(cpl):
                        if rover_distances[i] < obs_rad:
                            observer_count += 1

                    # Update POI observers
                    if observer_count >= cpl:
                        summed_observer_distances = sum(rover_distances[0:cpl])
                        counterfactual_global_reward += poi[poi_id, 2] / (summed_observer_distances / cpl)

                # Calculate D++ reward with n counterfactuals added
                temp_dpp = (counterfactual_global_reward - global_reward) / n_counters
                if temp_dpp > dpp_rewards[agent_id]:
                    dpp_rewards[agent_id] = temp_dpp
                    c = cpl + 1  # Stop iterrating
        else:
            dpp_rewards[agent_id] = d_rewards[agent_id]  # Returns difference reward

    return dpp_rewards

