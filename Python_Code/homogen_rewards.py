import numpy as np
import math
from parameters import Parameters as p


# DIFFERENCE REWARDS --------------------------------------------------------------------------------------------------
def calc_difference(rover_paths, poi_values, poi_positions, global_reward):
    number_agents = p.num_rovers
    number_pois = p.num_pois
    total_steps = p.num_steps + 1
    min_obs_distance = p.activation_dist
    inf = 1000.00

    difference_rewards = [0.0 for i in range(number_agents)]

    for agent_id in range(number_agents):

        for step_index in range(total_steps):
            counterfactual_global_reward = 0.0

            for poi_id in range(number_pois):
                # Count how many agents observe poi, update closest distance if necessary
                observer_count = 0
                observer_distances = [0.0 for i in range(number_agents)]
                summed_observer_distances = 0.0

                for other_agent_id in range(number_agents):

                    if agent_id != other_agent_id:  # Ignore self (Null Action)
                        # Calculate separation distance between poi and agent
                        x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                        y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                        distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                        if distance < p.min_distance:
                            distance = p.min_distance

                        observer_distances[other_agent_id] = distance

                        # Check if agent observes poi
                        if distance < min_obs_distance:
                            observer_count += 1
                    else:
                        observer_distances[other_agent_id] = inf  # Ignore self


                # update closest distance only if poi is observed
                if observer_count >= p.coupling:
                    for observer_id in range(p.coupling):
                        summed_observer_distances += min(observer_distances)
                        od_index = observer_distances.index(min(observer_distances))
                        observer_distances[od_index] = inf
                    counterfactual_global_reward += poi_values[poi_id] / ((1/p.coupling)*summed_observer_distances)

            temp_difference_reward = global_reward - counterfactual_global_reward
            if temp_difference_reward > difference_rewards[agent_id]:
                difference_rewards[agent_id] = temp_difference_reward

    return difference_rewards


# D++ REWARD ----------------------------------------------------------------------------------------------------------
def calc_dpp(rover_paths, poi_values, poi_positions, global_reward):
    number_agents = p.num_rovers
    number_pois = p.num_pois
    total_steps = p.num_steps + 1
    min_obs_distance = p.activation_dist
    inf = 1000.00

    difference_rewards = calc_difference(rover_paths, poi_values, poi_positions, global_reward)
    dpp_rewards = [0.0 for i in range(number_agents)]

    # Calculate Dpp Reward with (TotalAgents - 1) Counterfactuals
    n_counters = p.coupling-1
    for agent_id in range(number_agents):

        for step_index in range(total_steps):
            counterfactual_global_reward = 0.0

            for poi_id in range(number_pois):

                # Count how many agents observe poi, update closest distance if necessary
                observer_count = 0
                summed_observer_distances = 0.0
                observer_distances = [0.0 for i in range(number_agents + n_counters)]

                for other_agent_id in range(number_agents):
                    # Calculate separation distance between poi and agent
                    x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                    y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                    distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                    if distance < p.min_distance:
                        distance = p.min_distance

                    observer_distances[other_agent_id] = distance

                    if distance < min_obs_distance:
                        observer_count += 1

                # Add in counterfactual partners
                for partner_id in range(n_counters):
                    observer_distances[number_agents + partner_id] = observer_distances[agent_id]
                    if observer_distances[agent_id] < min_obs_distance:
                        observer_count += 1

                # update closest distance only if poi is observed
                if observer_count >= p.coupling:
                    for observer_id in range(p.coupling):
                        summed_observer_distances += min(observer_distances)
                        od_index = observer_distances.index(min(observer_distances))
                        observer_distances[od_index] = inf
                    counterfactual_global_reward += poi_values[poi_id] / (
                                (1 / p.coupling) * summed_observer_distances)

            temp_dpp_reward = (counterfactual_global_reward - global_reward) / (1 + n_counters)

            if dpp_rewards[agent_id] < temp_dpp_reward:
                dpp_rewards[agent_id] = temp_dpp_reward

    for agent_id in range(number_agents):

        if dpp_rewards[agent_id] > difference_rewards[agent_id]:
            for n_counters in range(p.coupling):
                if n_counters == 0: continue

                for step_index in range(total_steps):
                    counterfactual_global_reward = 0.0
                    for poi_id in range(number_pois):

                        # Count how many agents observe poi, update closest distance if necessary
                        observer_count = 0
                        summed_observer_distances = 0.0
                        observer_distances = [0.0 for i in range(number_agents + n_counters)]

                        for other_agent_id in range(number_agents):
                            # Calculate separation distance between poi and agent
                            x_distance = poi_positions[poi_id, 0] - rover_paths[step_index, other_agent_id, 0]
                            y_distance = poi_positions[poi_id, 1] - rover_paths[step_index, other_agent_id, 1]
                            distance = math.sqrt((x_distance * x_distance) + (y_distance * y_distance))

                            if distance < p.min_distance:
                                distance = p.min_distance

                            observer_distances[other_agent_id] = distance

                            if distance < min_obs_distance:
                                observer_count += 1

                        # Add in counterfactual partners
                        for partner_id in range(n_counters):
                            observer_distances[number_agents + partner_id] = observer_distances[agent_id]
                            if observer_distances[agent_id] < min_obs_distance:
                                observer_count += 1

                        # update closest distance only if poi is observed
                        if observer_count >= p.coupling:
                            for observer_id in range(p.coupling):
                                summed_observer_distances += min(observer_distances)
                                od_index = observer_distances.index(min(observer_distances))
                                observer_distances[od_index] = inf
                            counterfactual_global_reward += poi_values[poi_id] / ((1/p.coupling) * summed_observer_distances)

                    temp_dpp_reward = (counterfactual_global_reward - global_reward)/(1 + n_counters)

                    if dpp_rewards[agent_id] < temp_dpp_reward:
                        dpp_rewards[agent_id] = temp_dpp_reward
        else:
            dpp_rewards[agent_id] = difference_rewards[agent_id]  # Returns difference reward

    return dpp_rewards

# cpdef calc_sdpp(rover_path, poi_values, poi_positions):
#     cdef int nrovers = int(p.num_rovers*p.num_types)
#     cdef int npois = int(p.num_pois)
#     cdef int coupling = int(p.coupling)
#     cdef int poi_id, step_number, rover_id, rv, observer_count, od_index, other_rover_id, c
#     cdef double min_dist = p.min_distance
#     cdef double act_dist = p.activation_dist
#     cdef double rover_x_dist, rover_y_dist, distance, summed_distances, current_poi_reward, temp_reward, g_without_self
#     cdef double self_x, self_y, self_dist
#     cdef int num_steps = int(p.num_steps + 1)
#     cdef double inf = 1000.00
#     cdef double g_reward = 0.0
#     cdef double[:] difference_rewards = np.zeros(nrovers)
#     cdef double[:] dplusplus_reward = np.zeros(nrovers)
#
#     # CALCULATE GLOBAL REWARD
#     g_reward = calc_global(rover_path, poi_values, poi_positions)
#
#     # CALCULATE DIFFERENCE REWARD
#     dplusplus_reward = calc_difference(rover_path, poi_values, poi_positions)
#
#     # CALCULATE DPP REWARD
#     for c_count in range(coupling):
#
#         # Calculate Difference with Extra Me Reward
#         for rover_id in range(nrovers):
#             g_with_counterfactuals = 0.0
#
#             for poi_id in range(npois):
#                 current_poi_reward = 0.0
#
#                 for step_number in range(num_steps):
#                     observer_count = 0  # Track number of POI observers at time step
#                     observer_distances = []
#                     summed_distances = 0.0 # Denominator of reward function
#                     self_x = poi_positions[poi_id, 0] - rover_path[step_number, rover_id, 0]
#                     self_y = poi_positions[poi_id, 1] - rover_path[step_number, rover_id, 1]
#                     self_dist = math.sqrt((self_x**2) + (self_y**2))
#
#                     # Calculate distance between poi and agent
#                     for other_rover_id in range(nrovers):
#                         rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, other_rover_id, 0]
#                         rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, other_rover_id, 1]
#                         distance = math.sqrt((rover_x_dist**2) + (rover_y_dist**2))
#
#                         if distance <= min_dist:
#                             distance = min_dist
#                         observer_distances.append(distance)
#
#                         # Update observer count
#                         if distance <= act_dist:
#                             observer_count += 1
#
#                     if self_dist <= act_dist:  # Add Counterfactual Suggestions
#                         for c in range(c_count):
#                             if npois == 2:
#                                 observer_distances.append(two_poi_case_study(rover_id, poi_id, self_dist))
#                             if npois == 4:
#                                 observer_distances.append(four_corners_case_study(rover_id, poi_id, self_dist))
#
#                         observer_count += c_count
#
#                     if observer_count >= coupling:  # If coupling satisfied, compute reward
#                         for rv in range(coupling):
#                             summed_distances += min(observer_distances)
#                             od_index = observer_distances.index(min(observer_distances))
#                             observer_distances[od_index] = inf
#                         if summed_distances == 0:
#                             summed_distances = -1
#                         temp_reward = poi_values[poi_id]/summed_distances
#                     else:
#                         temp_reward = 0.0
#
#                     if temp_reward > current_poi_reward:
#                         current_poi_reward = temp_reward
#
#                 g_with_counterfactuals += current_poi_reward
#
#             temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + c_count)
#             if temp_dpp_reward > dplusplus_reward[rover_id]:
#                 dplusplus_reward[rover_id] = temp_dpp_reward
#
#     return dplusplus_reward
