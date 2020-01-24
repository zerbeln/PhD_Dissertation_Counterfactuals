import numpy as np
import math
from supervisor import get_counterfactual_partners, get_counterfactual_action

cpdef calc_global(object p, double [:, :, :] rover_paths, double[:, :] pois):
    """
    Calculate the global reward for the entire rover trajectory
    :param rover_paths:
    :param poi_values:
    :param poi_positions:
    :return: global_reward
    """
    cdef int poi_id, step_index, agent_id, observer, observer_count, od_index
    cdef int nrovers = int(p.num_rovers)
    cdef int npoi = int(p.num_pois)
    cdef double summed_observer_distances, x_distance, y_distance, distance
    cdef int cpl = int(p.coupling)
    cdef double cpl_double = p.coupling
    cdef int total_steps = int(p.num_steps + 1)  # The +1 is to account for the initial position
    cdef double inf = 1000.00
    cdef double min_dist = p.min_distance
    cdef double min_obs_dist = p.min_observation_dist
    cdef double global_reward = 0.0

    cdef double [:] rover_distances
    cdef double [:, :] poi_observer_distances = np.zeros((npoi, total_steps))
    cdef double [:] poi_observed = np.zeros(npoi)

    for poi_id in range(npoi):
        for step_index in range(total_steps):
            observer_count = 0
            rover_distances = np.zeros(nrovers)

            for agent_id in range(nrovers):
                # Calculate distance between agent and POI
                x_distance = pois[poi_id, 0] - rover_paths[step_index, agent_id, 0]
                y_distance = pois[poi_id, 1] - rover_paths[step_index, agent_id, 1]
                distance = math.sqrt((x_distance**2) + (y_distance**2))

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
            global_reward += pois[poi_id, 2] / (min(poi_observer_distances[poi_id])/cpl_double)

    return global_reward


# DIFFERENCE REWARDS --------------------------------------------------------------------------------------------------
cpdef calc_difference(object p, double [:, :, :] rover_paths, double [:, :] pois, double global_reward):
    """
    Calcualte each rover's difference reward from entire rover trajectory
    :param rover_paths:
    :param poi_values:
    :param poi_positions:
    :param global_reward:
    :return: difference_rewards (np array of size (n_rovers))
    """
    cdef int nrovers = int(p.num_rovers)
    cdef int npoi = int(p.num_pois)
    cdef int cpl = int(p.coupling)
    cdef int total_steps = int(p.num_steps + 1)  # The +1 is to account for the initial position
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
    cdef double [:] poi_observed

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
                        x_distance = pois[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                        y_distance = pois[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                        distance = math.sqrt((x_distance**2) + (y_distance**2))

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
                counterfactual_global_reward += pois[poi_id, 2] / (min(poi_observer_distances[poi_id])/cpl_double)
        difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards


# S-Difference REWARD -------------------------------------------------------------------------------------------------
cpdef calc_sd_reward(object p, double [:, :, :] rover_paths, double [:, :] pois, double global_reward):
    """
    Calcualte each rover's difference reward from entire rover trajectory
    :param rover_paths:
    :param poi_values:
    :param poi_positions:
    :param global_reward:
    :return: difference_rewards (np array of size (n_rovers))
    """
    cdef int nrovers = int(p.num_rovers)
    cdef int npoi = int(p.num_pois)
    cdef int cpl = int(p.coupling)
    cdef int total_steps = int(p.num_steps + 1)  # The +1 is to account for the initial position
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
    cdef double [:] poi_observed

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
                        x_distance = pois[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                        y_distance = pois[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                        distance = math.sqrt((x_distance**2) + (y_distance**2))

                        if distance < min_dist:
                            distance = min_dist

                        rover_distances[other_agent_id] = distance

                        # Check if agent observes poi
                        if distance < min_obs_distance:
                            observer_count += 1
                    else:
                        x_distance = pois[poi_id, 0] - rover_paths[step_index, agent_id, 0]
                        y_distance = pois[poi_id, 1] - rover_paths[step_index, agent_id, 1]
                        distance = math.sqrt((x_distance**2) + (y_distance**2))

                        rover_distances[agent_id] = get_counterfactual_action(distance, poi_id, pois)

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
                counterfactual_global_reward += pois[poi_id, 2] / (min(poi_observer_distances[poi_id])/cpl_double)
        difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards


# D++ REWARD ----------------------------------------------------------------------------------------------------------
cpdef calc_dpp(object p, double [:, :, :] rover_paths, double [:, :] pois, double global_reward):
    """
    Calculate D++ rewards for each rover across entire trajectory
    :param sgst: 
    :param rover_paths:
    :param poi_values:
    :param poi_positions:
    :param global_reward:
    :return: dpp_rewards (np array of size (n_rovers))
    """
    cdef int nrovers = int(p.num_rovers)
    cdef int npoi = int(p.num_pois)
    cdef int cpl = int(p.coupling)
    cdef int total_steps = int(p.num_steps + 1)  # The +1 is to account for the initial position (step 0)
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
    cdef double [:] poi_observed

    difference_rewards = calc_difference(p, rover_paths, pois, global_reward)
    dpp_rewards = np.zeros(nrovers)

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    n_counters = cpl - 1
    for agent_id in range(nrovers):
        poi_observer_distances = np.zeros((npoi, total_steps))
        poi_observed = np.zeros(npoi)

        for poi_id in range(npoi):
            for step_index in range(total_steps):
                observer_count = 0
                rover_distances = np.zeros(nrovers + n_counters)

                # Calculate linear distances between POI and agents, count observers
                for other_agent_id in range(nrovers):
                    x_distance = pois[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                    y_distance = pois[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                    distance = math.sqrt((x_distance**2) + (y_distance**2))

                    if distance < min_dist:
                        distance = min_dist

                    rover_distances[other_agent_id] = distance

                    if distance < min_obs_distance:
                        observer_count += 1

                # Create n counterfactual partners
                for partner_id in range(n_counters):
                    rover_distances[nrovers + partner_id] = rover_distances[agent_id]

                    if rover_distances[agent_id] < min_obs_distance:
                        observer_count += 1

                # Update POI observers
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
                counterfactual_global_reward += pois[poi_id, 2]/(min(poi_observer_distances[poi_id])/cpl_double)
        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / n_counters

    for agent_id in range(nrovers):
        if abs(dpp_rewards[agent_id]) > difference_rewards[agent_id]:
            dpp_rewards[agent_id] = 0.0
            poi_observer_distances = np.zeros((npoi, total_steps))
            poi_observed = np.zeros(npoi)

            for n_counters in range(cpl):
                if n_counters == 0:  # 0 counterfactual partnrs is identical to G
                    n_counters = 1
                for poi_id in range(npoi):
                    for step_index in range(total_steps):
                        observer_count = 0
                        rover_distances = np.zeros(nrovers + n_counters)

                        # Calculate linear distances between POI and agents, count observers
                        for other_agent_id in range(nrovers):
                            x_distance = pois[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                            y_distance = pois[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                            distance = math.sqrt((x_distance**2) + (y_distance**2))

                            if distance < min_dist:
                                distance = min_dist

                            rover_distances[other_agent_id] = distance

                            if distance < min_obs_distance:
                                observer_count += 1

                        # Create n counterfactual partners
                        for partner_id in range(n_counters):
                            rover_distances[nrovers + partner_id] = rover_distances[agent_id]

                            if rover_distances[agent_id] < min_obs_distance:
                                observer_count += 1

                        # Update POI observers
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

                # Calculate D++ reward with n counterfactuals added
                counterfactual_global_reward = 0.0
                for poi_id in range(npoi):
                    if poi_observed[poi_id] == 1:
                        counterfactual_global_reward += pois[poi_id, 2]/(min(poi_observer_distances[poi_id])/cpl_double)
                dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward)/n_counters
                if dpp_rewards[agent_id] > difference_rewards[agent_id]:
                    n_counters = cpl + 1  # Stop iterrating
        else:
            dpp_rewards[agent_id] = difference_rewards[agent_id]  # Returns difference reward

    return dpp_rewards

# S-D++ REWARD ----------------------------------------------------------------------------------------------------------
cpdef calc_sdpp(object p, double [:, :, :] rover_paths, double [:, :] pois, double global_reward, str sgst):
    """
    Calculate S-D++ rewards for each rover across entire trajectory
    :param sgst: 
    :param rover_paths:
    :param poi_values:
    :param poi_positions:
    :param global_reward:
    :return: dpp_rewards (np array of size (n_rovers))
    """
    cdef int nrovers = int(p.num_rovers)
    cdef int npoi = int(p.num_pois)
    cdef int cpl = int(p.coupling)
    cdef int total_steps = int(p.num_steps + 1)  # The +1 is to account for the initial position (step 0)
    cdef double cpl_double = p.coupling
    cdef double min_obs_distance = p.min_observation_dist
    cdef double min_dist = p.min_distance
    cdef double inf = 1000.00
    cdef int agent_id, poi_id, other_agent_id, observer_count, od_index, observer, n_counters, partner_id, step_index
    cdef double x_distance, y_distance, distance, summed_observer_distances, temp_dpp_reward
    cdef double counterfactual_global_reward
    cdef str suggestion

    cdef double [:] difference_rewards = np.zeros(nrovers)
    cdef double [:] dpp_rewards = np.zeros(nrovers)
    cdef double [:] observer_distances
    cdef double [:] counterfactual_agents
    cdef double [:, :] poi_observer_distances
    cdef double [:] poi_observed

    difference_rewards = calc_difference(p, rover_paths, pois, global_reward)
    dpp_rewards = np.zeros(nrovers)

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    n_counters = cpl - 1
    for agent_id in range(nrovers):
        poi_observer_distances = np.zeros((npoi, total_steps))
        poi_observed = np.zeros(npoi)

        for poi_id in range(npoi):
            suggestion = sgst
            for step_index in range(total_steps):
                observer_count = 0
                rover_distances = np.zeros(nrovers + n_counters)

                if p.step_suggestion_switch and p.step_switch_point == step_index:
                    suggestion = p.new_suggestion

                # Calculate linear distances between POI and agents, count observers
                for other_agent_id in range(nrovers):
                    x_distance = pois[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                    y_distance = pois[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                    distance = math.sqrt((x_distance**2) + (y_distance**2))

                    if distance < min_dist:
                        distance = min_dist

                    rover_distances[other_agent_id] = distance

                    if distance < min_obs_distance:
                        observer_count += 1

                # Get suggestion from supervisor if rover has discovered a POI
                if rover_distances[agent_id] <= min_obs_distance:
                    counterfactual_agents = get_counterfactual_partners(n_counters, nrovers, agent_id, rover_distances[agent_id], rover_paths, poi_id, pois, step_index, suggestion, min_dist, min_obs_distance)
                    for partner_id in range(n_counters):
                        rover_distances[nrovers + partner_id] = counterfactual_agents[partner_id]

                        if abs(counterfactual_agents[partner_id]) < min_obs_distance:
                            observer_count += 1

                # Update POI observers
                if observer_count >= cpl:
                    summed_observer_distances = 0.0
                    poi_observed[poi_id] = 1
                    for observer in range(cpl):  # Sum distances of closest observers
                        summed_observer_distances += min(rover_distances)
                        od_index = np.argmin(rover_distances)
                        rover_distances[od_index] = inf
                    if summed_observer_distances == 0.0:
                        summed_observer_distances = -1.0
                    poi_observer_distances[poi_id, step_index] = summed_observer_distances
                else:
                    poi_observer_distances[poi_id, step_index] = inf

        # Calculate D++ reward with n counterfactuals added
        counterfactual_global_reward = 0.0
        for poi_id in range(npoi):
            if poi_observed[poi_id] == 1:
                counterfactual_global_reward += pois[poi_id, 2]/(min(poi_observer_distances[poi_id])/cpl_double)
        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / n_counters

    for agent_id in range(nrovers):
        if abs(dpp_rewards[agent_id]) > difference_rewards[agent_id]:
            dpp_rewards[agent_id] = 0.0
            poi_observer_distances = np.zeros((npoi, total_steps))
            poi_observed = np.zeros(npoi)

            for n_counters in range(cpl):
                if n_counters == 0:  # 0 counterfactual partnrs is identical to G
                    n_counters = 1
                for poi_id in range(npoi):
                    suggestion = sgst
                    for step_index in range(total_steps):
                        observer_count = 0
                        rover_distances = np.zeros(nrovers + n_counters)

                        if p.step_suggestion_switch and p.step_switch_point == step_index and p.reward_type == "SDPP":
                            suggestion = p.new_suggestion

                        # Calculate linear distances between POI and agents, count observers
                        for other_agent_id in range(nrovers):
                            x_distance = pois[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                            y_distance = pois[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                            distance = math.sqrt((x_distance**2) + (y_distance**2))

                            if distance < min_dist:
                                distance = min_dist

                            rover_distances[other_agent_id] = distance

                            if distance < min_obs_distance:
                                observer_count += 1

                        # Get suggestion from supervisor if rover has discovered a POI
                        if rover_distances[agent_id] <= min_obs_distance:
                            counterfactual_agents = get_counterfactual_partners(n_counters, nrovers, agent_id, rover_distances[agent_id], rover_paths, poi_id, pois, step_index, suggestion, min_dist, min_obs_distance)
                            for partner_id in range(n_counters):
                                rover_distances[nrovers + partner_id] = counterfactual_agents[partner_id]

                                if abs(counterfactual_agents[partner_id]) < min_obs_distance:
                                    observer_count += 1

                        # Update POI observers
                        if observer_count >= cpl:
                            summed_observer_distances = 0.0
                            poi_observed[poi_id] = 1
                            for observer in range(cpl):  # Sum distances of closest observers
                                summed_observer_distances += min(rover_distances)
                                od_index = np.argmin(rover_distances)
                                rover_distances[od_index] = inf
                            if summed_observer_distances == 0.0:
                                summed_observer_distances = -1.0
                            poi_observer_distances[poi_id, step_index] = summed_observer_distances
                        else:
                            poi_observer_distances[poi_id, step_index] = inf

                # Calculate D++ reward with n counterfactuals added
                counterfactual_global_reward = 0.0
                for poi_id in range(npoi):
                    if poi_observed[poi_id] == 1:
                        counterfactual_global_reward += pois[poi_id, 2]/(min(poi_observer_distances[poi_id])/cpl_double)
                dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward)/n_counters
                if dpp_rewards[agent_id] > difference_rewards[agent_id]:
                    n_counters = cpl + 1  # Stop iterrating
        else:
            dpp_rewards[agent_id] = difference_rewards[agent_id]  # Returns difference reward

    return dpp_rewards
