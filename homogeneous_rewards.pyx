import numpy as np
from parameters import Parameters as p
import math

# GLOBAL REWARDS ------------------------------------------------------------------------------------------------------
cpdef calc_global(rover_history, poi_vals, poi_pos):
    cdef int number_agents = p.num_rovers
    cdef int number_pois = p.num_pois
    cdef double min_dist = p.min_distance
    cdef int total_steps = p.num_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] rover_path = rover_history
    cdef double[:] poi_values = poi_vals
    cdef double[:, :] poi_positions = poi_pos
    cdef int poi_id, step_number, agent_id, observer_count, od_index
    cdef double agent_x_dist, agent_y_dist, distance
    cdef double inf = 1000.0
    cdef double g_reward = 0.0 # Global reward
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double summed_distances = 0.0
    
    # For all POIs
    for poi_id in range(number_pois):
        current_poi_reward = 0.0

        # For all timesteps (rover steps)
        for step_number in range(total_steps):
            observer_count = 0
            observer_distances = np.zeros(number_agents)
            summed_distances = 0.0
            temp_reward = 0.0

            # For all agents
            # Calculate distance between poi and agent
            for agent_id in range(number_agents):
                agent_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, agent_id, 0]
                agent_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, agent_id, 1]
                distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                if distance <= min_dist:
                    distance = min_dist
                observer_distances[agent_id] = distance
                
                # Check if agent observes poi
                if distance <= activation_dist: # Rover is in observation range
                    observer_count += 1

            # update closest distance only if poi is observed    
            if observer_count >= coupling:
                for rv in range(coupling):
                    summed_distances += np.min(observer_distances[:])
                    assert(np.min(observer_distances[:]) <= activation_dist)
                    od_index = np.where(observer_distances[:] == np.min(observer_distances[:]))[0][0]
                    observer_distances[od_index] = inf
                temp_reward = poi_values[poi_id]/summed_distances
            else:
                temp_reward = 0.0

            if temp_reward > current_poi_reward:
                current_poi_reward = temp_reward

        g_reward += current_poi_reward

    return g_reward


# DIFFERENCE REWARDS -------------------------------------------------------------------------------------------------
cpdef calc_difference(rover_history, poi_vals, poi_pos):
    cdef int number_agents = p.num_rovers
    cdef int number_pois = p.num_pois
    cdef double min_dist = p.min_distance
    cdef int total_steps = p.num_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] rover_path = rover_history
    cdef double[:] poi_values = poi_vals
    cdef double[:, :] poi_positions = poi_pos
    cdef int poi_id, step_number, agent_id, observer_count, other_agent_id, od_index
    cdef double agent_x_dist, agent_y_dist, distance
    cdef double inf = 1000.0
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double summed_distances = 0.0
    cdef double[:] difference_rewards = np.zeros(number_agents)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global(rover_history, poi_vals, poi_pos)

    # CALCULATE DIFFERENCE REWARD
    for agent_id in range(number_agents):
        g_without_self = 0.0

        for poi_id in range(number_pois):
            current_poi_reward = 0.0

            for step_number in range(total_steps):
                observer_count = 0
                observer_distances = np.zeros(number_agents)
                summed_distances = 0.0
                temp_reward = 0.0

                # Calculate distance between poi and agent
                for other_agent_id in range(number_agents):
                    if agent_id != other_agent_id:
                        agent_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, other_agent_id, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, other_agent_id, 1]
                        distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                        if distance <= min_dist:
                            distance = min_dist
                        observer_distances[other_agent_id] = distance
                        
                        # Check if agent observes poi, update closest step distance
                        if distance < activation_dist:
                            observer_count += 1
                    else:
                        observer_distances[agent_id] = inf
                            
                # update closest distance only if poi is observed    
                if observer_count >= coupling:
                    for rv in range(coupling):
                        summed_distances += np.min(observer_distances[:])
                        assert(np.min(observer_distances[:]) <= activation_dist)
                        od_index = np.where(observer_distances[:] == np.min(observer_distances[:]))[0][0]
                        observer_distances[od_index] = inf
                    temp_reward = poi_values[poi_id]/summed_distances
                else:
                    temp_reward = 0.0

                if temp_reward > current_poi_reward:
                    current_poi_reward = temp_reward

            g_without_self += current_poi_reward
        difference_rewards[agent_id] = g_reward - g_without_self

    return difference_rewards


# D++ REWARD ----------------------------------------------------------------------------------------------------------
cpdef calc_dpp(rover_history, poi_vals, poi_pos):
    cdef int number_agents = p.num_rovers
    cdef int number_pois = p.num_pois
    cdef double min_dist = p.min_distance
    cdef int total_steps = p.num_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] rover_path = rover_history
    cdef double[:] poi_values = poi_vals
    cdef double[:, :] poi_positions = poi_pos
    cdef int poi_id, step_number, agent_id, observer_count, other_agent_id, counterfactual_count, od_index, rov_id
    cdef double agent_x_dist, agent_y_dist, distance
    cdef double inf = 1000.00
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double g_with_counterfactuals = 0.0 # Reward with n counterfactual partners added
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double temp_dpp_reward = 0.0
    cdef double summed_distances = 0.0
    cdef double[:] dplusplus_reward = np.zeros(number_agents)
    cdef double[:] difference_reward = np.zeros(number_agents)
    
    # CALCULATE GLOBAL REWARD
    g_reward = calc_global(rover_history, poi_vals, poi_pos)

    # CALCULATE DIFFERENCE REWARD
    difference_reward = calc_difference(rover_history, poi_vals, poi_pos)
    
    # CALCULATE DPP REWARD
    for counterfactual_count in range(coupling):

        # Calculate Difference with Extra Me Reward
        for agent_id in range(number_agents):
            g_with_counterfactuals = 0.0
            self_dist = 0.0

            for poi_id in range(number_pois):
                current_poi_reward = 0.0

                for step_number in range(total_steps):
                    observer_count = 0
                    observer_distances = []
                    summed_distances = 0.0
                    temp_reward = 0.0

                    # Calculate distance between poi and agent
                    for other_agent_id in range(number_agents):
                        agent_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, other_agent_id, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, other_agent_id, 1]
                        distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                        if distance < min_dist:
                            distance = min_dist
                        observer_distances.append(distance)

                        if other_agent_id == agent_id:
                            self_dist = distance # Track distance from self for counterfactuals

                        # Check if agent observes poi, update closest step distance
                        if distance <= activation_dist:
                            observer_count += 1

                    assert(len(observer_distances) == number_agents)
                    if observer_count < coupling:
                        if self_dist <= activation_dist:
                            for c in range(counterfactual_count):
                                observer_distances.append(self_dist)
                            observer_count += counterfactual_count

                    if observer_count >= coupling:
                        for rv in range(coupling):
                            summed_distances += min(observer_distances[:])
                            assert(np.min(observer_distances[:]) <= activation_dist)
                            od_index = observer_distances.index(min(observer_distances[:]))
                            observer_distances[od_index] = inf
                        temp_reward = poi_values[poi_id]/summed_distances
                    else:
                        temp_reward = 0.0

                    if temp_reward > current_poi_reward:
                        current_poi_reward = temp_reward

                g_with_counterfactuals += current_poi_reward

            temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + counterfactual_count)
            if temp_dpp_reward > dplusplus_reward[agent_id]:
                dplusplus_reward[agent_id] = temp_dpp_reward

    for rov_id in range(number_agents):
        if difference_reward[rov_id] > dplusplus_reward[rov_id]:
            dplusplus_reward[rov_id] = difference_reward[rov_id]

    return dplusplus_reward
