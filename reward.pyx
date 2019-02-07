import numpy as np
from parameters import Parameters as p
import math
from supervisor import closest_others, random_partners


cpdef rearrange_dist_vec(rov_distances):  # Rearrange distances from least to greatest
    cdef int size = len(rov_distances)
    for i in range(size):
        j = i + 1
        while j < size:
            if rov_distances[j] < rov_distances[i]:
                rov_distances[i], rov_distances[j] = rov_distances[j], rov_distances[i]
            j += 1

# GLOBAL REWARDS ------------------------------------------------------------------------------------------------------
cpdef calc_global_reward(rover_history, poi_vals, poi_pos):
    cdef int n_rovers = p.num_rovers
    cdef int n_pois = p.num_pois
    cdef double min_dist = p.min_distance
    cdef int num_steps = p.num_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] agent_pos_history = rover_history
    cdef double[:] poi_values = poi_vals
    cdef double[:, :] poi_positions = poi_pos
    cdef int poi_id, step_number, agent_id, observer_count
    cdef double agent_x_dist, agent_y_dist, distance
    cdef double inf = 1000000
    cdef double g_reward = 0.0 # Global reward
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double summed_distances = 0.0
    
    # For all POIs
    for poi_id in range(n_pois):
        current_poi_reward = 0.0

        # For all timesteps (rover steps)
        for step_number in range(num_steps):
            # Count how many agents observe poi, update closest distance if necessary
            observer_count = 0
            observer_distances = [0.0 for i in range(n_rovers)]
            summed_distances = 0.0
            temp_reward = 0.0

            # Calculate distance between poi and agent
            for agent_id in range(n_rovers):
                agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, agent_id, 0]
                agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, agent_id, 1]
                distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                if distance < min_dist:
                    distance = min_dist
                observer_distances[agent_id] = distance
                
                # Check if agent observes poi
                if distance <= activation_dist: # Rover is in observation range
                    observer_count += 1

            rearrange_dist_vec(observer_distances)

            # update closest distance only if poi is observed    
            if observer_count >= coupling:
                for rv in range(coupling):
                    summed_distances += observer_distances[rv]
                temp_reward = poi_values[poi_id]/summed_distances
            else:
                temp_reward = 0.0

            if temp_reward > current_poi_reward:
                current_poi_reward = temp_reward

        g_reward += current_poi_reward

    return g_reward


# DIFFERENCE REWARDS -------------------------------------------------------------------------------------------------
cpdef calc_difference_reward(rover_history, poi_vals, poi_pos):
    cdef int n_rovers = p.num_rovers
    cdef int n_pois = p.num_pois
    cdef double min_dist = p.min_distance
    cdef int num_steps = p.num_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] agent_pos_history = rover_history
    cdef double[:] poi_values = poi_vals
    cdef double[:, :] poi_positions = poi_pos
    cdef int poi_id, step_number, agent_id, observer_count, other_agent_id
    cdef double agent_x_dist, agent_y_dist, distance
    cdef double inf = 1000000
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double summed_distances = 0.0
    cdef double[:] difference_reward = np.zeros(n_rovers)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global_reward(rover_history, poi_vals, poi_pos)

    # CALCULATE DIFFERENCE REWARD
    for agent_id in range(n_rovers):
        g_without_self = 0.0

        for poi_id in range(n_pois):
            current_poi_reward = 0.0

            for step_number in range(num_steps):
                # Count how many agents observe poi, update closest distance if necessary
                observer_count = 0
                observer_distances = [0.0 for i in range(n_rovers)]
                summed_distances = 0.0
                temp_reward = 0.0

                # Calculate distance between poi and agent
                for other_agent_id in range(n_rovers):
                    if agent_id != other_agent_id:
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, other_agent_id, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, other_agent_id, 1]
                        distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                        if distance < min_dist:
                            distance = min_dist
                        observer_distances[other_agent_id] = distance

                        # Check if agent observes poi, update closest step distance
                        if distance <= activation_dist:
                            observer_count += 1
                    else:
                        observer_distances[agent_id] = inf

                rearrange_dist_vec(observer_distances)

                # update closest distance only if poi is observed
                if observer_count >= coupling:
                    for rv in range(coupling):
                        summed_distances += observer_distances[rv]
                    temp_reward = poi_values[poi_id]/summed_distances
                else:
                    temp_reward = 0.0

                if temp_reward > current_poi_reward:
                    current_poi_reward = temp_reward

            g_without_self += current_poi_reward
        difference_reward[agent_id] = g_reward - g_without_self

    return difference_reward


# D++ REWARD ----------------------------------------------------------------------------------------------------------
cpdef calc_dpp_reward(rover_history, poi_vals, poi_pos):
    cdef int n_rovers = p.num_rovers
    cdef int n_pois = p.num_pois
    cdef double min_dist = p.min_distance
    cdef int num_steps = p.num_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] agent_pos_history = rover_history
    cdef double[:] poi_values = poi_vals
    cdef double[:, :] poi_positions = poi_pos
    cdef int poi_id, step_number, agent_id, observer_count, other_agent_id, counterfactual_count
    cdef double agent_x_dist, agent_y_dist, distance
    cdef double inf = 1000000
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double g_with_counterfactuals = 0.0 # Reward with n counterfactual partners added
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double temp_dpp_reward = 0.0
    cdef double summed_distances = 0.0
    cdef double[:] dplusplus_reward = np.zeros(n_rovers)
    cdef double[:] difference_reward = np.zeros(n_rovers)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global_reward(rover_history, poi_vals, poi_pos)

    # CALCULATE DIFFERENCE REWARD
    difference_reward = calc_difference_reward(rover_history, poi_vals, poi_pos)

    # CALCULATE DPP REWARD
    for counterfactual_count in range(coupling):

        # Calculate Difference with Extra Me Reward
        for agent_id in range(n_rovers):
            g_with_counterfactuals = 0.0
            self_dist = 0.0

            for poi_id in range(n_pois):
                current_poi_reward = 0.0

                for step_number in range(num_steps):
                    # Count how many agents observe poi, update closest distance if necessary
                    observer_count = 0
                    observer_distances = [0.0 for i in range(n_rovers)]
                    summed_distances = 0.0
                    temp_reward = 0.0

                    # Calculate distance between poi and agent
                    for other_agent_id in range(n_rovers):
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, other_agent_id, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, other_agent_id, 1]
                        distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                        if distance < min_dist:
                            distance = min_dist
                        observer_distances[other_agent_id] = distance

                        if other_agent_id == agent_id:
                            self_dist = distance # Track distance from self for counterfactuals

                        # Check if agent observes poi, update closest step distance
                        if distance <= activation_dist:
                            observer_count += 1

                    if observer_count < coupling:
                        if self_dist <= activation_dist:
                            for c in range(counterfactual_count):
                                observer_distances.append(self_dist)
                            observer_count += counterfactual_count

                    rearrange_dist_vec(observer_distances)

                    # update closest distance only if poi is observed
                    if observer_count >= coupling:
                        for rv in range(coupling):
                            summed_distances += observer_distances[rv]
                        temp_reward = poi_values[poi_id]/summed_distances
                    else:
                        temp_reward = 0.0

                    if temp_reward > current_poi_reward:
                        current_poi_reward = temp_reward

                g_with_counterfactuals += current_poi_reward

            temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + counterfactual_count)
            if temp_dpp_reward > dplusplus_reward[agent_id]:
                dplusplus_reward[agent_id] = temp_dpp_reward

    for id in range(n_rovers):
        if difference_reward[id] > dplusplus_reward[id]:
            dplusplus_reward[id] = difference_reward[id]

    return dplusplus_reward

# S-D++ REWARD --------------------------------------------------------------------------------------------------------
cpdef calc_sdpp_reward(rover_history, poi_vals, poi_pos):
    cdef int n_rovers = p.num_rovers
    cdef int n_pois = p.num_pois
    cdef double min_dist = p.min_distance
    cdef int num_steps = p.num_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] agent_pos_history = rover_history
    cdef double[:] poi_values = poi_vals
    cdef double[:, :] poi_positions = poi_pos
    cdef double[:] rov_partners
    cdef int poi_id, step_number, agent_id, observer_count, other_agent_id, counterfactual_count
    cdef double agent_x_dist, agent_y_dist, distance
    cdef double inf = 1000000
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double g_with_counterfactuals = 0.0 # Reward with n counterfactual partners added
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double temp_dpp_reward = 0.0
    cdef double summed_distances = 0.0
    cdef double[:] difference_reward = np.zeros(n_rovers)
    cdef double[:] dplusplus_reward = np.zeros(n_rovers)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global_reward(rover_history, poi_vals, poi_pos)

    # CALCULATE DIFFERENCE REWARD
    difference_reward = calc_difference_reward(rover_history, poi_vals, poi_pos)

    # CALCULATE S-DPP REWARD
    for counterfactual_count in range(coupling):

        # Calculate reward with suggested counterfacual partners
        for agent_id in range(n_rovers):
            g_with_counterfactuals = 0.0
            self_dist = 0.0

            for poi_id in range(n_pois):
                current_poi_reward = 0.0

                for step_number in range(num_steps):
                    # Count how many agents observe poi, update closest distance if necessary
                    observer_count = 0
                    observer_distances = [0.0 for i in range(n_rovers)]
                    summed_distances = 0.0
                    temp_reward = 0.0

                    # Calculate distance between poi and agent
                    for other_agent_id in range(n_rovers):
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[step_number, other_agent_id, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[step_number, other_agent_id, 1]
                        distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                        if distance < min_dist:
                            distance = min_dist
                        observer_distances[other_agent_id] = distance

                        if other_agent_id == agent_id:
                            self_dist = distance # Track distance from self for counterfactuals

                        # Check if agent observes poi, update closest step distance
                        if distance <= activation_dist:
                            observer_count += 1

                    if observer_count < coupling:  # Suggest counterfactual partners
                        rov_partners = closest_others(step_number, counterfactual_count, agent_id, poi_id, rover_history, poi_pos)
                        # rov_partners = random_partners(step_number, counterfactual_count, agent_id, poi_id, rover_history, poi_pos)

                        for rovid in range(counterfactual_count):
                            observer_distances.append(rov_partners[rovid])  # Append n closest
                        observer_count += counterfactual_count

                    rearrange_dist_vec(observer_distances)  # Rearrange rover distances with added counterfactuals

                    # update closest distance only if poi is observed
                    if observer_count >= coupling:
                        for rv in range(coupling):
                            summed_distances += observer_distances[rv]
                        temp_reward = poi_values[poi_id]/(0.5*summed_distances)
                    else:
                        temp_reward = 0.0

                    if temp_reward > current_poi_reward:
                        current_poi_reward = temp_reward

                g_with_counterfactuals += current_poi_reward

            temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + counterfactual_count)
            if temp_dpp_reward > dplusplus_reward[agent_id]:
                dplusplus_reward[agent_id] = temp_dpp_reward

    for id in range(n_rovers):
        if difference_reward[id] > dplusplus_reward[id]:
            dplusplus_reward[id] = difference_reward[id]

    return dplusplus_reward
