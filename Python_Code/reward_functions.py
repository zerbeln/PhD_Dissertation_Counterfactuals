import numpy as np
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

        counterfactual_global_reward = 0.0
        for poi_id in range(n_poi):  # For each POI
            observer_count = 0
            rover_distances = observer_distances[poi_id].copy()
            rover_distances[agent_id] = 1000.00
            rover_distances = np.sort(rover_distances)

            for i in range(cpl):
                dist = rover_distances[i]
                if dist < obs_rad:
                    observer_count += 1

            # Compute reward if coupling is satisfied
            if observer_count >= cpl:
                summed_observer_distances = 0.0
                for i in range(cpl):  # Sum distances of closest observers
                    summed_observer_distances += rover_distances[i]

                counterfactual_global_reward += poi[poi_id, 2] / (summed_observer_distances/cpl)

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
    for agent_id in range(n_rovers):

        counterfactual_global_reward = 0.0
        for poi_id in range(n_poi):  # For each POI
            rover_distances = observer_distances[poi_id].copy()
            rover_distances[agent_id] = 1000.00  # Create counterfactual action for agent i

            best_dist = min(rover_distances)
            if best_dist < obs_rad:
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
        counterfactual_global_reward = 0.0
        for poi_id in range(n_poi):
            observer_count = 0
            rover_distances = observer_distances[poi_id].copy()
            counterfactual_rovers = np.ones(n_counters) * observer_distances[poi_id, agent_id]
            rover_distances = np.append(rover_distances, counterfactual_rovers)
            rover_distances = np.sort(rover_distances)

            for i in range(cpl):
                if rover_distances[i] < obs_rad:
                    observer_count += 1

            # Update POI observers
            if observer_count >= cpl:
                summed_observer_distances = 0.0
                for i in range(cpl):  # Sum distances of closest observers
                    summed_observer_distances += rover_distances[i]
                counterfactual_global_reward += poi[poi_id, 2]/(summed_observer_distances/cpl)

        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / n_counters

    for agent_id in range(n_rovers):
        if dpp_rewards[agent_id] > d_rewards[agent_id]:
            dpp_rewards[agent_id] = 0.0

            for c in range(cpl-1):
                n_counters = c + 1
                counterfactual_global_reward = 0.0
                for poi_id in range(n_poi):
                    observer_count = 0
                    rover_distances = observer_distances[poi_id].copy()
                    counterfactual_rovers = np.ones(n_counters) * observer_distances[poi_id, agent_id]
                    rover_distances = np.append(rover_distances, counterfactual_rovers)
                    rover_distances = np.sort(rover_distances)

                    for i in range(cpl):
                        if rover_distances[i] < obs_rad:
                            observer_count += 1

                    # Update POI observers
                    if observer_count >= cpl:
                        summed_observer_distances = 0.0
                        for i in range(cpl):  # Sum distances of closest observers
                            summed_observer_distances += rover_distances[i]
                        counterfactual_global_reward += poi[poi_id, 2] / (summed_observer_distances / cpl)

                # Calculate D++ reward with n counterfactuals added
                temp_dpp = (counterfactual_global_reward - global_reward)/n_counters
                if temp_dpp > dpp_rewards[agent_id]:
                    dpp_rewards[agent_id] = temp_dpp
                    c = cpl + 1  # Stop iterrating
        else:
            dpp_rewards[agent_id] = d_rewards[agent_id]  # Returns difference reward

    return dpp_rewards

