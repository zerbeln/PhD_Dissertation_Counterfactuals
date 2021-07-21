import numpy as np
import math
from parameters import parameters as p

# DIFFERENCE REWARDS --------------------------------------------------------------------------------------------------
def calc_difference_tight(observer_distances, poi, global_reward):
    """
    Calcualte each rover's difference reward from entire rover trajectory
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
        poi_observer_distances = np.zeros(n_poi)  # Tracks summed observer distances
        poi_observed = np.zeros(n_poi)

        for poi_id in range(n_poi):  # For each POI
            observer_count = 0
            rover_distances = np.zeros(n_rovers)  # Track distances between rovers and POI

            # Count how many agents observe poi, update closest distances
            for other_agent_id in range(n_rovers):
                if agent_id != other_agent_id:  # Remove current rover's trajectory
                    dist = observer_distances[poi_id, other_agent_id]
                    rover_distances[other_agent_id] = dist

                    if dist < obs_rad:
                        observer_count += 1
                else:
                    rover_distances[agent_id] = 1000.00  # Ignore self

            # Determine if coupling is satisfied
            if observer_count >= cpl:
                summed_observer_distances = 0.0
                poi_observed[poi_id] = 1

                rover_distances = np.sort(rover_distances)  # Sort from least to greatest
                for i in range(cpl):  # Sum distances of closest observers
                    summed_observer_distances += rover_distances[i]
                poi_observer_distances[poi_id] = summed_observer_distances

        counterfactual_global_reward = 0.0
        for poi_id in range(n_poi):
            if poi_observed[poi_id] == 1:
                counterfactual_global_reward += poi[poi_id, 2] / (poi_observer_distances[poi_id]/cpl)
        difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards


def calc_difference_loose(observer_distances, poi, global_reward):
    """
    Calcualte each rover's difference reward from entire rover trajectory
    :param observer_distances: Each rover's distance to each POI
    :param poi: np array with X-Y coordinates and value for each POI
    :param global_reward: Reward given to the team from the world
    :return: difference_rewards (np array of size (n_rovers))
    """

    obs_rad = p["observation_radius"]
    n_poi = p["n_poi"]
    n_rovers = p["n_rovers"]

    difference_rewards = np.zeros(n_rovers)
    for agent_id in range(n_rovers):  # For each rover
        poi_observed = np.zeros(n_poi)

        for poi_id in range(n_poi):  # For each POI
            rover_distances = observer_distances[poi_id].copy()
            rover_distances[agent_id] = 1000.00  # Ignore self
            # Check for observers other than the current agent
            for other_agent_id in range(n_rovers):
                if agent_id != other_agent_id:  # Remove current rover's trajectory
                    dist = observer_distances[poi_id, other_agent_id]

                    # Determine if coupling is satisfied
                    if dist < obs_rad:
                        poi_observed[poi_id] = 1
                        break

            counterfactual_global_reward = 0.0
            if poi_observed[poi_id] == 1:
                counterfactual_global_reward += poi[poi_id, 2] / min(rover_distances)
            difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards


# D++ REWARD ----------------------------------------------------------------------------------------------------------
def calc_dpp(observer_distances, poi, global_reward):
    """
    Calculate D++ rewards for each rover across entire trajectory
    :param observer_distances: Each rover's distance to each POI
    :param poi: np array with X-Y coordinates and value for each POI
    :param global_reward: Reward given to the team from the world
    :return: dpp_rewards (np array of size (n_rovers))
    """

    n_poi = p["n_poi"]
    n_rovers = p["n_rovers"]
    cpl = p["coupling"]
    obs_rad = p["observation_radius"]
    d_rewards = calc_difference_tight(observer_distances, poi, global_reward)
    dpp_rewards = np.zeros(n_rovers)

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    n_counters = cpl - 1
    for agent_id in range(n_rovers):
        poi_observer_distances = np.zeros(n_poi)
        poi_observed = np.zeros(n_poi)

        for poi_id in range(n_poi):
            observer_count = 0
            rover_distances = np.zeros(n_rovers + n_counters)

            # Calculate linear distances between POI and agents, count observers
            for other_agent_id in range(n_rovers):
                dist = observer_distances[poi_id, other_agent_id]
                rover_distances[other_agent_id] = dist
                if dist < obs_rad:
                    observer_count += 1

            # Create n counterfactual partners
            for partner_id in range(n_counters):
                rover_distances[n_rovers + partner_id] = rover_distances[agent_id]

                if rover_distances[agent_id] < obs_rad:
                    observer_count += 1

            # Update POI observers
            if observer_count >= cpl:
                summed_observer_distances = 0.0
                poi_observed[poi_id] = 1
                rover_distances = np.sort(rover_distances)
                for i in range(cpl):  # Sum distances of closest observers
                    summed_observer_distances += rover_distances[i]
                poi_observer_distances[poi_id] = summed_observer_distances

        counterfactual_global_reward = 0.0
        for poi_id in range(n_poi):
            if poi_observed[poi_id] == 1:
                counterfactual_global_reward += poi[poi_id, 2]/(poi_observer_distances[poi_id]/cpl)
        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / n_counters

    for agent_id in range(n_rovers):
        if dpp_rewards[agent_id] > d_rewards[agent_id]:
            dpp_rewards[agent_id] = 0.0
            poi_observer_distances = np.zeros(n_poi)
            poi_observed = np.zeros(n_poi)

            for c in range(cpl-1):
                n_counters = c+1
                for poi_id in range(n_poi):
                    observer_count = 0
                    rover_distances = np.zeros(n_rovers + n_counters)
                    # Calculate linear distances between POI and agents, count observers
                    for other_agent_id in range(n_rovers):
                        dist = observer_distances[poi_id, other_agent_id]
                        rover_distances[other_agent_id] = dist
                        if dist < obs_rad:
                            observer_count += 1

                    # Create n counterfactual partners
                    for partner_id in range(n_counters):
                        rover_distances[n_rovers + partner_id] = rover_distances[agent_id]

                        if rover_distances[agent_id] < obs_rad:
                            observer_count += 1

                    # Update POI observers
                    if observer_count >= cpl:
                        summed_observer_distances = 0.0
                        poi_observed[poi_id] = 1
                        rover_distances = np.sort(rover_distances)
                        for i in range(cpl):  # Sum distances of closest observers
                            summed_observer_distances += rover_distances[i]
                        poi_observer_distances[poi_id] = summed_observer_distances

                # Calculate D++ reward with n counterfactuals added
                counterfactual_global_reward = 0.0
                for poi_id in range(n_poi):
                    if poi_observed[poi_id] == 1:
                        counterfactual_global_reward += poi[poi_id, 2]/(poi_observer_distances[poi_id]/cpl)
                temp_dpp = (counterfactual_global_reward - global_reward)/n_counters
                if temp_dpp > d_rewards[agent_id] and temp_dpp > dpp_rewards[agent_id]:
                    dpp_rewards[agent_id] = temp_dpp
                    c = cpl + 1  # Stop iterrating
        else:
            dpp_rewards[agent_id] = d_rewards[agent_id]  # Returns difference reward

    return dpp_rewards


def target_poi(target, observer_distances, pois, rover_id):
    """
    Provides agents with rewards for observing a specific, target POI
    """

    # Parameters
    obs_rad = p["observation_radius"]

    # Variables
    reward = 0.0
    observed = 0

    # Calculate distance between agent and POI
    dist = observer_distances[target, rover_id]

    # Check if agent observes poi and update observer count if true
    if dist < obs_rad:
        observed = 1
        reward = pois[target, 2]/dist

    return reward, observed


def travel_in_direction(direction, rover_positions, rover_id):
    reward = 0
    n_rovers = p["n_rovers"]
    world_x = p["x_dim"]
    world_y = p["y_dim"]

    for rover_id in range(n_rovers):
        rover_x = rover_positions[rover_id, 0]
        rover_y = rover_positions[rover_id, 1]

        if direction == 1:  # West
            reward += world_x - rover_x
        elif direction == 2:  # East
            reward += rover_x
        elif direction == 3:  # North
            reward += world_y - rover_y
        elif direction == 4:  # South
            reward += rover_y

    return reward
