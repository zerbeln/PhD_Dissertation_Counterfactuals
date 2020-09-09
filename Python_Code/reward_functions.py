import numpy as np
import math
from parameters import parameters as p

# GLOBAL REWARD -------------------------------------------------------------------------------------------------------
def calc_global(rover_paths, poi):
    """
    Calculate the global reward for the entire rover trajectory
    :param rover_paths: X-Y coordinates of each rover at each time step
    :param poi: np array with X-Y coordinates and value for each POI
    :return: global_reward
    """

    total_steps = int(p["n_steps"] + 1)  # The +1 is to account for the initial position
    inf = 1000.00
    global_reward = 0.0

    poi_observer_distances = np.zeros((p["n_poi"], total_steps))  # Tracks summed observer distances
    poi_observed = np.zeros(p["n_poi"])
    for poi_id in range(p["n_poi"]):
        for step_index in range(total_steps):
            observer_count = 0
            rover_distances = np.zeros(p["n_rovers"])

            for agent_id in range(p["n_rovers"]):
                # Calculate distance between agent and POI
                x_distance = poi[poi_id, 0] - rover_paths[step_index, agent_id, 0]
                y_distance = poi[poi_id, 1] - rover_paths[step_index, agent_id, 1]
                distance = math.sqrt((x_distance**2) + (y_distance**2))

                if distance < p["min_distance"]:
                    distance = p["min_distance"]

                rover_distances[agent_id] = distance

                # Check if agent observes poi and update observer count if true
                if distance < p["min_obs_dist"]:
                    observer_count += 1

            # Update global reward if POI is observed
            if observer_count >= p["coupling"]:
                poi_observed[poi_id] = 1
                summed_observer_distances = 0.0
                for observer in range(p["coupling"]):  # Sum distances of closest observers
                    summed_observer_distances += min(rover_distances)
                    od_index = np.argmin(rover_distances)
                    rover_distances[od_index] = inf
                poi_observer_distances[poi_id, step_index] = summed_observer_distances
            else:
                poi_observer_distances[poi_id, step_index] = inf

    for poi_id in range(p["n_poi"]):
        if poi_observed[poi_id] == 1:
            global_reward += poi[poi_id, 2] / (min(poi_observer_distances[poi_id])/p["coupling"])

    return global_reward


# DIFFERENCE REWARDS --------------------------------------------------------------------------------------------------
def calc_difference(rover_paths, poi, global_reward):
    """
    Calcualte each rover's difference reward from entire rover trajectory
    :param rover_paths: X-Y coordinates of each rover at each time step
    :param poi: np array with X-Y coordinates and value for each POI
    :param global_reward: Reward given to the team from the world
    :return: difference_rewards (np array of size (n_rovers))
    """

    total_steps = int(p["n_steps"] + 1)  # The +1 is to account for the initial position
    inf = 1000.00
    difference_rewards = np.zeros(p["n_rovers"])


    for agent_id in range(p["n_rovers"]):  # For each rover
        poi_observer_distances = np.zeros((p["n_poi"], total_steps))  # Tracks summed observer distances
        poi_observed = np.zeros(p["n_poi"])

        for poi_id in range(p["n_poi"]):  # For each POI
            for step_index in range(total_steps):  # For each step in trajectory
                observer_count = 0
                rover_distances = np.zeros(p["n_rovers"])  # Track distances between rovers and POI

                # Count how many agents observe poi, update closest distances
                for other_agent_id in range(p["n_rovers"]):
                    if agent_id != other_agent_id:  # Remove current rover's trajectory
                        # Calculate separation distance between poi and agent
                        x_distance = poi[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                        y_distance = poi[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                        distance = math.sqrt((x_distance**2) + (y_distance**2))

                        if distance < p["min_distance"]:
                            distance = p["min_distance"]

                        rover_distances[other_agent_id] = distance

                        # Check if agent observes poi
                        if distance < p["min_obs_distance"]:
                            observer_count += 1
                    else:
                        rover_distances[agent_id] = inf  # Ignore self

                # Determine if coupling is satisfied
                if observer_count >= p["coupling"]:
                    summed_observer_distances = 0.0
                    poi_observed[poi_id] = 1
                    for observer in range(p["coupling"]):  # Sum distances of closest observers
                        summed_observer_distances += min(rover_distances)
                        od_index = np.argmin(rover_distances)
                        rover_distances[od_index] = inf
                    poi_observer_distances[poi_id, step_index] = summed_observer_distances
                else:
                    poi_observer_distances[poi_id, step_index] = inf

        counterfactual_global_reward = 0.0
        for poi_id in range(p["n_poi"]):
            if poi_observed[poi_id] == 1:
                counterfactual_global_reward += poi[poi_id, 2] / (min(poi_observer_distances[poi_id])/p["coupling"])
        difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards

# D++ REWARD ----------------------------------------------------------------------------------------------------------
def calc_dpp(rover_paths, poi, global_reward):
    """
    Calculate D++ rewards for each rover across entire trajectory
    :param rover_paths: X-Y coordinates of each rover at each time step
    :param poi: np array with X-Y coordinates and value for each POI
    :param global_reward: Reward given to the team from the world
    :return: dpp_rewards (np array of size (n_rovers))
    """
    total_steps = int(p["n_steps"] + 1)  # The +1 is to account for the initial position (step 0)
    inf = 1000.00
    difference_rewards = calc_difference(rover_paths, poi, global_reward)
    dpp_rewards = np.zeros(p["n_rovers"])

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    n_counters = p["coupling"] - 1
    for agent_id in range(p["n_rovers"]):
        poi_observer_distances = np.zeros((p["n_poi"], total_steps))
        poi_observed = np.zeros(p["n_poi"])

        for poi_id in range(p["n_poi"]):
            for step_index in range(total_steps):
                observer_count = 0
                rover_distances = np.zeros(p["n_rovers"] + n_counters)

                # Calculate linear distances between POI and agents, count observers
                for other_agent_id in range(p["n_rovers"]):
                    x_distance = poi[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                    y_distance = poi[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                    distance = math.sqrt((x_distance**2) + (y_distance**2))

                    if distance < p["min_distance"]:
                        distance = p["min_distance"]

                    rover_distances[other_agent_id] = distance

                    if distance < p["min_obs_distance"]:
                        observer_count += 1

                # Create n counterfactual partners
                for partner_id in range(n_counters):
                    rover_distances[p["n_rovers"] + partner_id] = rover_distances[agent_id]

                    if rover_distances[agent_id] < p["min_obs_distance"]:
                        observer_count += 1

                # Update POI observers
                if observer_count >= p["coupling"]:
                    summed_observer_distances = 0.0
                    poi_observed[poi_id] = 1
                    for observer in range(p["coupling"]):  # Sum distances of closest observers
                        summed_observer_distances += min(rover_distances)
                        od_index = np.argmin(rover_distances)
                        rover_distances[od_index] = inf
                    poi_observer_distances[poi_id, step_index] = summed_observer_distances
                else:
                    poi_observer_distances[poi_id, step_index] = inf

        counterfactual_global_reward = 0.0
        for poi_id in range(p["n_poi"]):
            if poi_observed[poi_id] == 1:
                counterfactual_global_reward += poi[poi_id, 2]/(min(poi_observer_distances[poi_id])/p["coupling"])
        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / n_counters

    for agent_id in range(p["n_rovers"]):
        if abs(dpp_rewards[agent_id]) > difference_rewards[agent_id]:
            dpp_rewards[agent_id] = 0.0
            poi_observer_distances = np.zeros((p["n_poi"], total_steps))
            poi_observed = np.zeros(p["n_poi"])

            for n_counters in range(p["coupling"]):
                if n_counters == 0:  # 0 counterfactual partnrs is identical to G
                    n_counters = 1
                for poi_id in range(p["n_poi"]):
                    for step_index in range(total_steps):
                        observer_count = 0
                        rover_distances = np.zeros(p["n_rovers"] + n_counters)

                        # Calculate linear distances between POI and agents, count observers
                        for other_agent_id in range(p["n_rovers"]):
                            x_distance = poi[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                            y_distance = poi[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                            distance = math.sqrt((x_distance**2) + (y_distance**2))

                            if distance < p["min_distance"]:
                                distance = p["min_distance"]

                            rover_distances[other_agent_id] = distance

                            if distance < p["min_obs_distance"]:
                                observer_count += 1

                        # Create n counterfactual partners
                        for partner_id in range(n_counters):
                            rover_distances[p["n_rovers"] + partner_id] = rover_distances[agent_id]

                            if rover_distances[agent_id] < p["min_obs_distance"]:
                                observer_count += 1

                        # Update POI observers
                        if observer_count >= p["coupling"]:
                            summed_observer_distances = 0.0
                            poi_observed[poi_id] = 1
                            for observer in range(p["coupling"]):  # Sum distances of closest observers
                                summed_observer_distances += min(rover_distances)
                                od_index = np.argmin(rover_distances)
                                rover_distances[od_index] = inf
                            poi_observer_distances[poi_id, step_index] = summed_observer_distances
                        else:
                            poi_observer_distances[poi_id, step_index] = inf

                # Calculate D++ reward with n counterfactuals added
                counterfactual_global_reward = 0.0
                for poi_id in range(p["n_poi"]):
                    if poi_observed[poi_id] == 1:
                        counterfactual_global_reward += poi[poi_id, 2]/(min(poi_observer_distances[poi_id])/p["coupling"])
                dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward)/n_counters
                if dpp_rewards[agent_id] > difference_rewards[agent_id]:
                    n_counters = p["coupling"] + 1  # Stop iterrating
        else:
            dpp_rewards[agent_id] = difference_rewards[agent_id]  # Returns difference reward

    return dpp_rewards
