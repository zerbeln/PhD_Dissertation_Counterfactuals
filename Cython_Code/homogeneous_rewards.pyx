import numpy as np
from AADI_RoverDomain.parameters import Parameters as p
import math
from Cython_Code.suggestions import get_counterfactual_partners

cpdef calc_global(rover_paths, poi_values, poi_positions):
    """
    Calculate the global reward for the entire rover trajectory
    :param rover_paths:
    :param poi_values:
    :param poi_positions:
    :return: global_reward
    """
    cdef int poi_id, step_index, agent_id, observer, observer_count, od_index
    cdef int nrovers = p.num_rovers
    cdef int npoi = p.num_pois
    cdef double summed_observer_distances, x_distance, y_distance, distance
    cdef int cpl = p.coupling
    cdef int total_steps = p.num_steps + 1  # The +1 is to account for the initial position
    cdef double inf = 1000.00
    cdef double min_dist = p.min_distance
    cdef double min_obs_dist = p.min_observation_dist
    cdef double global_reward = 0.0

    cdef double [:] rover_distances
    cdef double [:, :] poi_observer_distances = np.zeros((npoi, total_steps))
    cdef int [:] poi_observed = np.zeros(npoi)

    for poi_id in range(npoi):
        for step_index in range(total_steps):
            observer_count = 0
            rover_distances = np.zeros(nrovers)

            for agent_id in range(nrovers):
                # Calculate distance between agent and POI
                x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, agent_id, 0]
                y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, agent_id, 1]
                distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                if distance < min_dist:
                    distance = min_dist

                rover_distances[agent_id] = distance

                # Check if agent observes poi and update observer count if true
                if distance < min_obs_dist:
                    observer_count += 1

            # Update global reward if POI is observed
            if observer_count >= cpl:
                poi_observed[poi_id] = 1
                summed_observer_distances = 0.0
                for observer in range(cpl):  # Sum distances of closest observers
                    summed_observer_distances += min(rover_distances)
                    od_index = np.argmin(rover_distances)
                    rover_distances[od_index] = inf
                poi_observer_distances[poi_id, step_index] = summed_observer_distances
            else:
                poi_observer_distances[poi_id, step_index] = inf

    for poi_id in range(npoi):
        if poi_observed[poi_id] == 1:
            global_reward += poi_values[poi_id]

    return global_reward


# DIFFERENCE REWARDS --------------------------------------------------------------------------------------------------
cpdef calc_difference(rover_paths, poi_values, poi_positions, global_reward):
    """
    Calcualte each rover's difference reward from entire rover trajectory
    :param rover_paths:
    :param poi_values:
    :param poi_positions:
    :param global_reward:
    :return: difference_rewards (np array of size (n_rovers))
    """
    cdef int nrovers = p.num_rovers
    cdef int npoi = p.num_pois
    cdef int cpl = p.coupling
    cdef int total_steps = p.num_steps + 1  # The +1 is to account for the initial position
    cdef double cpl_double = p.coupling
    cdef double min_dist = p.min_distance
    cdef double min_obs_distance = p.min_observation_dist
    cdef double inf = 1000.00
    cdef int agent_id, poi_id, other_agent_id, observer_count, od_index, observer, step_index
    cdef double x_distance, y_distance, distance, summed_observer_distances
    cdef double counterfactual_global_reward

    cdef double [:] difference_rewards = np.zeros(nrovers)
    cdef double [:] rover_distances
    cdef double [:, :] poi_observer_distances
    cdef int [:] poi_observed

    for agent_id in range(nrovers):  # For each rover
        poi_observer_distances = np.zeros((npoi, total_steps))  # Tracks summed observer distances
        poi_observed = np.zeros(npoi)

        for poi_id in range(npoi):  # For each POI
            for step_index in range(total_steps):  # For each step in trajectory
                observer_count = 0
                rover_distances = np.zeros(nrovers)  # Track distances between rovers and POI

                # Count how many agents observe poi, update closest distances
                for other_agent_id in range(nrovers):
                    if agent_id != other_agent_id:  # Remove current rover's trajectory
                        # Calculate separation distance between poi and agent
                        x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                        y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                        distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                        if distance < min_dist:
                            distance = min_dist

                        rover_distances[other_agent_id] = distance

                        # Check if agent observes poi
                        if distance < min_obs_distance:
                            observer_count += 1
                    else:
                        rover_distances[agent_id] = inf  # Ignore self

                # Determine if coupling is satisfied
                if observer_count >= cpl:
                    summed_observer_distances = 0.0
                    poi_observed[poi_id] = 1
                    for observer in range(cpl):  # Sum distances of closest observers
                        summed_observer_distances += min(rover_distances)
                        od_index = np.argmin(rover_distances)
                        rover_distances[od_index] = inf
                    poi_observer_distances[poi_id, step_index] = summed_observer_distances
                else:
                    poi_observer_distances[poi_id, step_index] = inf

        counterfactual_global_reward = 0.0
        for poi_id in range(npoi):
            if poi_observed[poi_id] == 1:
                counterfactual_global_reward += poi_values[poi_id] / (min(poi_observer_distances[poi_id])/cpl_double)
        difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards


# D++ REWARD ----------------------------------------------------------------------------------------------------------
cpdef calc_dpp(rover_paths, poi_values, poi_positions, global_reward):
    """
    Calculate D++ rewards for each rover across entire trajectory
    :param rover_paths:
    :param poi_values:
    :param poi_positions:
    :param global_reward:
    :return: dpp_rewards (np array of size (n_rovers))
    """
    cdef int nrovers = p.num_rovers
    cdef int npoi = p.num_pois
    cdef int cpl = p.coupling
    cdef int total_steps = p.num_steps + 1 # The +1 is to account for the initial position
    cdef double cpl_double = p.coupling
    cdef double min_obs_distance = p.min_observation_dist
    cdef double min_dist = p.min_distance
    cdef double inf = 1000.00
    cdef int agent_id, poi_id, other_agent_id, observer_count, od_index, observer, n_counters, partner_id, step_index
    cdef double x_distance, y_distance, distance, summed_observer_distances, temp_dpp_reward
    cdef double counterfactual_global_reward

    cdef double [:] difference_rewards = np.zeros(nrovers)
    cdef double [:] dpp_rewards = np.zeros(nrovers)
    cdef double [:] observer_distances
    cdef double [:] counterfactual_agents
    cdef double [:, :] poi_observer_distances
    cdef int [:] poi_observed

    difference_rewards = calc_difference(rover_paths, poi_values, poi_positions, global_reward)
    dpp_rewards = np.zeros(nrovers)

    # Calculate Dpp Reward with (TotalAgents - 1) Counterfactuals
    n_counters = cpl - 1
    for agent_id in range(nrovers):
        poi_observer_distances = np.zeros((npoi, total_steps))
        poi_observed = np.zeros(npoi)

        for poi_id in range(npoi):
            for step_index in range(total_steps):
                observer_count = 0
                rover_distances = np.zeros(nrovers + n_counters)

                # Count how many agents observe poi, update closest distances
                for other_agent_id in range(nrovers):
                    # Calculate separation distance between poi and agent
                    x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                    y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                    distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                    if distance < min_dist:
                        distance = min_dist

                    rover_distances[other_agent_id] = distance

                    if distance < min_obs_distance:
                        observer_count += 1

                # Add in counterfactual partners
                counterfactual_agents = get_counterfactual_partners(n_counters, agent_id, rover_distances[agent_id], rover_paths, poi_id, poi_values, step_index)
                for partner_id in range(n_counters):
                    rover_distances[nrovers + partner_id] = counterfactual_agents[partner_id]

                    if counterfactual_agents[partner_id] < min_obs_distance:
                        observer_count += 1

                # Update whether or not POI has been observed
                if observer_count >= cpl:
                    summed_observer_distances = 0.0
                    poi_observed[poi_id] = 1
                    for observer in range(cpl):  # Sum distances of closest observers
                        summed_observer_distances += min(rover_distances)
                        od_index = np.argmin(rover_distances)
                        rover_distances[od_index] = inf
                    poi_observer_distances[poi_id, step_index] = summed_observer_distances
                else:
                    poi_observer_distances[poi_id, step_index] = inf

        counterfactual_global_reward = 0.0
        for poi_id in range(npoi):
            if poi_observed[poi_id] == 1:
                counterfactual_global_reward += poi_values[poi_id]/(min(poi_observer_distances[poi_id])/cpl_double)
        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / n_counters

    for agent_id in range(nrovers):
        if dpp_rewards[agent_id] > difference_rewards[agent_id]:
            poi_observer_distances = np.zeros((npoi, total_steps))
            poi_observed = np.zeros(npoi)

            for n_counters in range(cpl-1):
                if n_counters == 0:  # 0 counterfactual partnrs is identical to G
                    n_counters = 1
                for poi_id in range(npoi):
                    for step_index in range(total_steps):
                        observer_count = 0
                        rover_distances = np.zeros(nrovers + n_counters)

                        # Count how many agents observe poi, update closest distances
                        for other_agent_id in range(nrovers):
                            # Calculate separation distance between poi and agent
                            x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                            y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                            distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                            if distance < min_dist:
                                distance = min_dist

                            rover_distances[other_agent_id] = distance

                            if distance < min_obs_distance:
                                observer_count += 1

                        # Add in counterfactual partners
                        counterfactual_agents = get_counterfactual_partners(n_counters, agent_id, rover_distances[agent_id], rover_paths, poi_id, poi_values, step_index)
                        for partner_id in range(n_counters):
                            rover_distances[nrovers + partner_id] = counterfactual_agents[partner_id]

                            if counterfactual_agents[partner_id] < min_obs_distance:
                                observer_count += 1

                        # Determine if coupling has been satisfied
                        if observer_count >= cpl:
                            summed_observer_distances = 0.0
                            poi_observed[poi_id] = 1
                            for observer in range(cpl):  # Sum distances of closest observers
                                summed_observer_distances += min(rover_distances)
                                od_index = np.argmin(rover_distances)
                                rover_distances[od_index] = inf
                            poi_observer_distances[poi_id, step_index] = summed_observer_distances
                        else:
                            poi_observer_distances[poi_id, step_index] = inf

                counterfactual_global_reward = 0.0
                for poi_id in range(npoi):
                    if poi_observed[poi_id] == 1:
                        counterfactual_global_reward += poi_values[poi_id]/(min(poi_observer_distances[poi_id])/cpl_double)
                temp_dpp_reward = (counterfactual_global_reward - global_reward)/n_counters
                if dpp_rewards[agent_id] < temp_dpp_reward:
                    dpp_rewards[agent_id] = temp_dpp_reward
        else:
            dpp_rewards[agent_id] = difference_rewards[agent_id]  # Returns difference reward

    return dpp_rewards
