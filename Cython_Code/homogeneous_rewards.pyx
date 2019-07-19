import numpy as np
from AADI_RoverDomain.parameters import Parameters as p
import math
from Python_Code.suggestions import partner_distance


# DIFFERENCE REWARDS --------------------------------------------------------------------------------------------------
cpdef calc_difference(rover_paths, poi_values, poi_positions, global_reward, step_index):
    cdef int number_agents = p.num_rovers
    cdef int number_pois = p.num_pois
    cdef int cpling = p.coupling
    cdef double cple = p.coupling
    cdef double min_obs_distance = p.min_observation_dist
    cdef double cutoff_distance = p.min_distance
    cdef double inf = 1000.00
    cdef int agent_id, poi_id, other_agent_id, observer_count, od_index, observer_id
    cdef double x_distance, y_distance, distance, summed_observer_distances, temp_difference_reward
    cdef double counterfactual_global_reward

    cdef double [:] difference_rewards = np.zeros(number_agents)
    cdef double [:] observer_distances

    for agent_id in range(number_agents):

        counterfactual_global_reward = 0.0

        for poi_id in range(number_pois):
            # Count how many agents observe poi, update closest distance if necessary
            observer_count = 0
            observer_distances = np.zeros(number_agents)
            summed_observer_distances = 0.0

            for other_agent_id in range(number_agents):

                if agent_id != other_agent_id:  # Ignore self (Null Action)
                    # Calculate separation distance between poi and agent
                    x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                    y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                    distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                    if distance < cutoff_distance:
                        distance = cutoff_distance

                    observer_distances[other_agent_id] = distance

                    # Check if agent observes poi
                    if distance < min_obs_distance:
                        observer_count += 1
                else:
                    observer_distances[other_agent_id] = inf  # Ignore self


            # Update reward if coupling is satisfied
            if observer_count >= cpling:
                for observer_id in range(cpling):
                    summed_observer_distances += min(observer_distances)
                    od_index = np.argmin(observer_distances)
                    observer_distances[od_index] = inf
                counterfactual_global_reward += poi_values[poi_id] / ((1/cple)*summed_observer_distances)

        temp_difference_reward = global_reward - counterfactual_global_reward
        if temp_difference_reward > difference_rewards[agent_id]:
            difference_rewards[agent_id] = temp_difference_reward

    return difference_rewards


# D++ REWARD ----------------------------------------------------------------------------------------------------------
cpdef calc_dpp(rover_paths, poi_values, poi_positions, global_reward, step_index):
    cdef int number_agents = p.num_rovers
    cdef int number_pois = p.num_pois
    cdef int cpling = p.coupling
    cdef double cple = p.coupling
    cdef double min_obs_distance = p.min_observation_dist
    cdef double cutoff_distance = p.min_distance
    cdef double inf = 1000.00
    cdef int agent_id, poi_id, other_agent_id, observer_count, od_index, observer_id, n_counters, partner_id
    cdef double x_distance, y_distance, distance, summed_observer_distances, temp_dpp_reward
    cdef double counterfactual_global_reward

    cdef double [:] difference_rewards = np.zeros(number_agents)
    cdef double [:] dpp_rewards = np.zeros(number_agents)
    cdef double [:] observer_distances

    difference_rewards = calc_difference(rover_paths, poi_values, poi_positions, global_reward, step_index)

    n_counters = cpling-1
    for agent_id in range(number_agents):

        counterfactual_global_reward = 0.0

        for poi_id in range(number_pois):

            # Count how many agents observe poi, update closest distance if necessary
            observer_count = 0
            summed_observer_distances = 0.0
            observer_distances = np.zeros(number_agents + n_counters)

            for other_agent_id in range(number_agents):
                # Calculate separation distance between poi and agent
                x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                if distance < cutoff_distance:
                    distance = cutoff_distance

                observer_distances[other_agent_id] = distance

                if distance < min_obs_distance:
                    observer_count += 1

            # Add in counterfactual partners
            for partner_id in range(n_counters):
                observer_distances[number_agents + partner_id] = observer_distances[agent_id]
                if observer_distances[agent_id] < min_obs_distance:
                    observer_count += 1

            # update closest distance only if poi is observed
            if observer_count >= cpling:
                for observer_id in range(cpling):
                    summed_observer_distances += min(observer_distances)
                    od_index = np.argmin(observer_distances)
                    observer_distances[od_index] = inf
                counterfactual_global_reward += poi_values[poi_id] / ((1 / cple) * summed_observer_distances)

        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / (1 + n_counters)

    for agent_id in range(number_agents):

        if dpp_rewards[agent_id] > difference_rewards[agent_id]:
            for n_counters in range(cpling):

                counterfactual_global_reward = 0.0

                for poi_id in range(number_pois):

                    # Count how many agents observe poi, update closest distance if necessary
                    observer_count = 0
                    summed_observer_distances = 0.0
                    observer_distances = np.zeros(number_agents + n_counters)

                    for other_agent_id in range(number_agents):
                        # Calculate separation distance between poi and agent
                        x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                        y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                        distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                        if distance < cutoff_distance:
                            distance = cutoff_distance

                        observer_distances[other_agent_id] = distance

                        if distance < min_obs_distance:
                            observer_count += 1

                    # Add in counterfactual partners
                    for partner_id in range(n_counters):
                        observer_distances[number_agents + partner_id] = observer_distances[agent_id]
                        if observer_distances[number_agents + partner_id] < min_obs_distance:
                            observer_count += 1

                    # update closest distance only if poi is observed
                    if observer_count >= cpling:
                        for observer_id in range(cpling):
                            summed_observer_distances += min(observer_distances)
                            od_index = np.argmin(observer_distances)
                            observer_distances[od_index] = inf
                        counterfactual_global_reward += poi_values[poi_id] / ((1/cple) * summed_observer_distances)

                temp_dpp_reward = (counterfactual_global_reward - global_reward)/(1 + n_counters)

                if dpp_rewards[agent_id] < temp_dpp_reward:
                    dpp_rewards[agent_id] = temp_dpp_reward
        else:
            dpp_rewards[agent_id] = difference_rewards[agent_id]  # Returns difference reward


    return dpp_rewards

# SD++ REWARD ----------------------------------------------------------------------------------------------------------
cpdef calc_sdpp(rover_paths, poi_values, poi_positions, global_reward, step_index):
    cdef int number_agents = p.num_rovers
    cdef int number_pois = p.num_pois
    cdef int cpling = p.coupling
    cdef double cple = p.coupling
    cdef double min_obs_distance = p.min_observation_dist
    cdef double cutoff_distance = p.min_distance
    cdef double inf = 1000.00
    cdef int agent_id, poi_id, other_agent_id, observer_count, od_index, observer_id, n_counters, partner_id, added_partners
    cdef double x_distance, y_distance, distance, summed_observer_distances, temp_dpp_reward, self_x, self_y
    cdef double counterfactual_global_reward

    cdef double [:] difference_rewards = np.zeros(number_agents)
    cdef double [:] dpp_rewards = np.zeros(number_agents)
    cdef double [:] suggested_partners
    cdef double [:] observer_distances

    difference_rewards = calc_difference(rover_paths, poi_values, poi_positions, global_reward, step_index)

    # Calculate Dpp Reward with (TotalAgents - 1) Counterfactuals
    n_counters = cpling-1
    for agent_id in range(number_agents):

        counterfactual_global_reward = 0.0

        for poi_id in range(number_pois):

            # Count how many agents observe poi, update closest distance if necessary
            observer_count = 0
            summed_observer_distances = 0.0
            observer_distances = np.zeros(number_agents + n_counters)

            for other_agent_id in range(number_agents):
                # Calculate separation distance between poi and agent
                x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                if distance < cutoff_distance:
                    distance = cutoff_distance

                observer_distances[other_agent_id] = distance

                if distance < min_obs_distance:
                    observer_count += 1

            # Add in counterfactual partners
            self_x = rover_paths[step_index, agent_id, 0]; self_y = rover_paths[step_index, agent_id, 1]
            suggested_partners, added_observers = partner_distance(n_counters, observer_distances[agent_id], agent_id, poi_id, poi_values)
            for partner_id in range(n_counters):
                observer_distances[number_agents + partner_id] = suggested_partners[partner_id]
            observer_count += added_observers

            # update closest distance only if poi is observed
            if observer_count >= cpling:
                for observer_id in range(cpling):
                    summed_observer_distances += min(observer_distances)
                    od_index = np.argmin(observer_distances)
                    observer_distances[od_index] = inf
                if summed_observer_distances == 0:
                    summed_observer_distances = -1
                counterfactual_global_reward += poi_values[poi_id] / ((1 / cple) * summed_observer_distances)

        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / (1 + n_counters)

    for agent_id in range(number_agents):

        if dpp_rewards[agent_id] > difference_rewards[agent_id]:
            for n_counters in range(cpling):

                counterfactual_global_reward = 0.0

                for poi_id in range(number_pois):

                    # Count how many agents observe poi, update closest distance if necessary
                    observer_count = 0
                    summed_observer_distances = 0.0
                    observer_distances = np.zeros(number_agents + n_counters)

                    for other_agent_id in range(number_agents):
                        # Calculate separation distance between poi and agent
                        x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                        y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                        distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                        if distance < cutoff_distance:
                            distance = cutoff_distance

                        observer_distances[other_agent_id] = distance

                        if distance < min_obs_distance:
                            observer_count += 1

                    # Add in counterfactual partners
                    self_x = rover_paths[step_index, agent_id, 0]; self_y = rover_paths[step_index, agent_id, 1]
                    suggested_partners, added_observers = partner_distance(n_counters, observer_distances[agent_id], agent_id, poi_id, poi_values)
                    for partner_id in range(n_counters):
                        observer_distances[number_agents + partner_id] = suggested_partners[partner_id]
                    observer_count += added_observers

                    # update closest distance only if poi is observed
                    if observer_count >= cpling:
                        for observer_id in range(cpling):
                            summed_observer_distances += min(observer_distances)
                            od_index = np.argmin(observer_distances)
                            observer_distances[od_index] = inf
                        if summed_observer_distances == 0:
                            summed_observer_distances = -1
                        counterfactual_global_reward += poi_values[poi_id] / ((1/cple) * summed_observer_distances)

                temp_dpp_reward = (counterfactual_global_reward - global_reward)/(1 + n_counters)

                if dpp_rewards[agent_id] < temp_dpp_reward:
                    dpp_rewards[agent_id] = temp_dpp_reward
        else:
            dpp_rewards[agent_id] = difference_rewards[agent_id]  # Returns difference reward


    return dpp_rewards
