import numpy as np
import math
from AADI_RoverDomain.parameters import Parameters as p
from Python_Code.suggestions import get_counterfactual_partners, get_cpartners_step_switch

# GLOBAL REWARDS ------------------------------------------------------------------------------------------------------
def calc_global(rover_paths, poi_values, poi_positions):
    """
    Calculate the global reward for the entire rover trajectory
    :param rover_paths:
    :param poi_values:
    :param poi_positions:
    :return: global_reward
    """
    total_steps = p.num_steps + 1  # The +1 is to account for the initial position
    inf = 1000.00
    global_reward = 0.0

    poi_observer_distances = np.zeros((p.num_pois, total_steps))
    poi_observed = [False for _ in range(p.num_pois)]
    for poi_id in range(p.num_pois):
        for step_index in range(total_steps):
            observer_count = 0
            rover_distances = np.zeros(p.num_rovers)

            for agent_id in range(p.num_rovers):
                # Calculate distance between agent and POI
                x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, agent_id, 0]
                y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, agent_id, 1]
                distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                if distance < p.min_distance:
                    distance = p.min_distance

                rover_distances[agent_id] = distance

                # Check if agent observes poi and update observer count if true
                if distance < p.min_observation_dist:
                    observer_count += 1

            # Update global reward if POI is observed
            if observer_count >= p.coupling:
                poi_observed[poi_id] = True
                summed_observer_distances = 0.0
                for observer in range(p.coupling):  # Sum distances of closest observers
                    summed_observer_distances += min(rover_distances)
                    od_index = np.argmin(rover_distances)
                    rover_distances[od_index] = inf
                poi_observer_distances[poi_id, step_index] = summed_observer_distances
            else:
                poi_observer_distances[poi_id, step_index] = inf

    for poi_id in range(p.num_pois):
        if poi_observed[poi_id] == True:
            global_reward += poi_values[poi_id]

    return global_reward



def calc_difference(rover_paths, poi_values, poi_positions, global_reward):
    """
    Calcualte each rover's difference reward from entire rover trajectory
    :param rover_paths:
    :param poi_values:
    :param poi_positions:
    :param global_reward:
    :return: difference_rewards (np array of size (n_rovers))
    """
    min_obs_distance = p.min_observation_dist
    total_steps = p.num_steps + 1  # The +1 is to account for the initial position
    inf = 1000.00

    difference_rewards = np.zeros(p.num_rovers)

    for agent_id in range(p.num_rovers):  # For each rover
        poi_observer_distances = np.zeros((p.num_pois, total_steps))  # Tracks summed observer distances
        poi_observed = [False for _ in range(p.num_pois)]

        for poi_id in range(p.num_pois):  # For each POI
            for step_index in range(total_steps):  # For each step in trajectory
                observer_count = 0
                rover_distances = np.zeros(p.num_rovers)  # Track distances between rovers and POI

                # Count how many agents observe poi, update closest distances
                for other_agent_id in range(p.num_rovers):
                    if agent_id != other_agent_id:  # Remove current rover's trajectory
                        # Calculate separation distance between poi and agent
                        x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                        y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                        distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                        if distance < p.min_distance:
                            distance = p.min_distance

                        rover_distances[other_agent_id] = distance

                        # Check if agent observes poi
                        if distance < min_obs_distance:
                            observer_count += 1
                    else:
                        rover_distances[agent_id] = inf  # Ignore self

                # Determine if coupling is satisfied
                if observer_count >= p.coupling:
                    summed_observer_distances = 0.0
                    poi_observed[poi_id] = True
                    for observer in range(p.coupling):  # Sum distances of closest observers
                        summed_observer_distances += min(rover_distances)
                        od_index = np.argmin(rover_distances)
                        rover_distances[od_index] = inf
                    poi_observer_distances[poi_id, step_index] = summed_observer_distances
                else:
                    poi_observer_distances[poi_id, step_index] = inf

        counterfactual_global_reward = 0.0
        for poi_id in range(p.num_pois):
            if poi_observed[poi_id] == True:
                counterfactual_global_reward += poi_values[poi_id] / (min(poi_observer_distances[poi_id])/p.coupling)
        difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards


def calc_dpp(rover_paths, poi_values, poi_positions, global_reward):
    """
    Calculate D++ rewards for each rover across entire trajectory
    :param rover_paths:
    :param poi_values:
    :param poi_positions:
    :param global_reward:
    :return: dpp_rewards (np array of size (n_rovers))
    """
    min_obs_distance = p.min_observation_dist
    total_steps = p.num_steps + 1  # The +1 is to account for the initial position
    inf = 1000.00

    difference_rewards = calc_difference(rover_paths, poi_values, poi_positions, global_reward)
    dpp_rewards = np.zeros(p.num_rovers)

    # Calculate Dpp Reward with (TotalAgents - 1) Counterfactuals
    n_counters = p.coupling - 1
    for agent_id in range(p.num_rovers):
        poi_observer_distances = np.zeros((p.num_pois, total_steps))
        poi_observed = [False for _ in range(p.num_pois)]

        for poi_id in range(p.num_pois):
            for step_index in range(total_steps):
                observer_count = 0
                rover_distances = np.zeros(p.num_rovers+n_counters)

                # Count how many agents observe poi, update closest distances
                for other_agent_id in range(p.num_rovers):
                    # Calculate separation distance between poi and agent
                    x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                    y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                    distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                    if distance < p.min_distance:
                        distance = p.min_distance

                    rover_distances[other_agent_id] = distance

                    if distance < min_obs_distance:
                        observer_count += 1

                # Add in counterfactual partners
                counterfactual_agents = get_counterfactual_partners(n_counters, agent_id, rover_distances[agent_id], rover_paths, poi_id, poi_values, poi_positions, step_index)
                # counterfactual_agents = get_cpartners_step_switch(n_counters, agent_id, rover_distances[agent_id], rover_paths, poi_id, poi_values, poi_positions, step_index)
                for partner_id in range(n_counters):
                    rover_distances[p.num_rovers+partner_id] = counterfactual_agents[partner_id]

                    if counterfactual_agents[partner_id] < min_obs_distance:
                        observer_count += 1

                # Update whether or not POI has been observed
                if observer_count >= p.coupling:
                    summed_observer_distances = 0.0
                    poi_observed[poi_id] = True
                    for observer in range(p.coupling):  # Sum distances of closest observers
                        summed_observer_distances += min(rover_distances)
                        od_index = np.argmin(rover_distances)
                        rover_distances[od_index] = inf
                    poi_observer_distances[poi_id, step_index] = summed_observer_distances
                else:
                    poi_observer_distances[poi_id, step_index] = inf

        counterfactual_global_reward = 0.0
        for poi_id in range(p.num_pois):
            if poi_observed[poi_id] == True:
                counterfactual_global_reward += poi_values[poi_id]/(min(poi_observer_distances[poi_id])/p.coupling)
        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / n_counters

    for agent_id in range(p.num_rovers):
        if dpp_rewards[agent_id] > difference_rewards[agent_id]:
            poi_observer_distances = np.zeros((p.num_pois, total_steps))
            poi_observed = [False for _ in range(p.num_pois)]

            for n_counters in range(p.coupling-1):
                if n_counters == 0:  # 0 counterfactual partnrs is identical to G
                    n_counters = 1
                for poi_id in range(p.num_pois):
                    for step_index in range(total_steps):
                        observer_count = 0
                        rover_distances = np.zeros(p.num_rovers + n_counters)

                        # Count how many agents observe poi, update closest distances
                        for other_agent_id in range(p.num_rovers):
                            # Calculate separation distance between poi and agent
                            x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                            y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                            distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                            if distance < p.min_distance:
                                distance = p.min_distance

                            rover_distances[other_agent_id] = distance

                            if distance < min_obs_distance:
                                observer_count += 1

                        # Add in counterfactual partners
                        counterfactual_agents = get_counterfactual_partners(n_counters, agent_id, rover_distances[agent_id], rover_paths, poi_id, poi_values, poi_positions, step_index)
                        # counterfactual_agents = get_cpartners_step_switch(n_counters, agent_id, rover_distances[agent_id], rover_paths, poi_id, poi_values, poi_positions, step_index)
                        for partner_id in range(n_counters):
                            rover_distances[p.num_rovers+partner_id] = counterfactual_agents[partner_id]

                            if counterfactual_agents[partner_id] < min_obs_distance:
                                observer_count += 1

                        # Determine if coupling has been satisfied
                        if observer_count >= p.coupling:
                            summed_observer_distances = 0.0
                            poi_observed[poi_id] = True
                            for observer in range(p.coupling):  # Sum distances of closest observers
                                summed_observer_distances += min(rover_distances)
                                od_index = np.argmin(rover_distances)
                                rover_distances[od_index] = inf
                            poi_observer_distances[poi_id, step_index] = summed_observer_distances
                        else:
                            poi_observer_distances[poi_id, step_index] = inf

                counterfactual_global_reward = 0.0
                for poi_id in range(p.num_pois):
                    if poi_observed[poi_id] == True:
                        counterfactual_global_reward += poi_values[poi_id]/(min(poi_observer_distances[poi_id])/p.coupling)
                temp_dpp_reward = (counterfactual_global_reward - global_reward)/n_counters
                if dpp_rewards[agent_id] < temp_dpp_reward:
                    dpp_rewards[agent_id] = temp_dpp_reward
        else:
            dpp_rewards[agent_id] = difference_rewards[agent_id]  # Returns difference reward

    return dpp_rewards
