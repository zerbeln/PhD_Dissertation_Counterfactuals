import numpy as np
cimport cython
from parameters import Parameters as p
from code.supervisor import closest_others, random_partners, value_based_partners
import math

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def rearrange_dist_vec(rov_distances):  # Rearrange distances from least to greatest
    cdef int size = len(rov_distances)
    for i in range(size):
        j = i + 1
        while j < size:
            if rov_distances[j] < rov_distances[i]:
                rov_distances[i], rov_distances[j] = rov_distances[j], rov_distances[i]
            j += 1

# GLOBAL REWARDS ------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def calc_global_reward(data):
    # Set parameters from those found in data dictionary
    cdef int number_agents = p.number_of_agents
    cdef int number_pois = p.number_of_pois
    cdef double min_dist = p.min_distance
    cdef int total_steps = p.total_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] agent_pos_history = data["Agent Position History"]
    cdef double[:] poi_values = data['Poi Values']
    cdef double[:, :] poi_positions = data["Poi Positions"]
    cdef int poi_id, step_number, agent_id, observer_count
    cdef double agent_x_dist, agent_y_dist, distance
    cdef double inf = float("inf")
    cdef double g_reward = 0.0 # Global reward
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double summed_distances = 0.0
    
    # For all POIs
    for poi_id in range(number_pois):
        current_poi_reward = 0.0

        # For all timesteps (rover steps)
        for step_number in range(total_steps):
            # Count how many agents observe poi, update closest distance if necessary
            observer_count = 0
            observer_distances = [0.0 for i in range(number_agents)]
            summed_distances = 0.0
            temp_reward = 0.0

            # For all agents
            # Calculate distance between poi and agent
            for agent_id in range(number_agents):
                agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[agent_id, step_number, 0]
                agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[agent_id, step_number, 1]
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
                    assert(observer_distances[rv] <= activation_dist)
                temp_reward = poi_values[poi_id]/summed_distances
            else:
                temp_reward = 0.0

            if temp_reward > current_poi_reward:
                current_poi_reward = temp_reward

        g_reward += current_poi_reward

    data["Global Reward"] = g_reward
    data["Agent Rewards"] = np.ones(number_agents) * g_reward


# DIFFERENCE REWARDS -------------------------------------------------------------------------------------------------
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def calc_difference_reward(data):
    cdef int number_agents = p.number_of_agents
    cdef int number_pois = p.number_of_pois
    cdef double min_dist = p.min_distance
    cdef int total_steps = p.total_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] agent_pos_history = data["Agent Position History"]
    cdef double[:] poi_values = data['Poi Values']
    cdef double[:, :] poi_positions = data["Poi Positions"]
    cdef int poi_id, step_number, agent_id, observer_count, other_agent_id
    cdef double agent_x_dist, agent_y_dist, distance
    cdef double inf = float("inf")
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double summed_distances = 0.0
    cdef double[:] difference_reward = np.zeros(number_agents)

    # CALCULATE GLOBAL REWARD
    # For all POIs
    for poi_id in range(number_pois):
        current_poi_reward = 0.0

        # For all timesteps (rover steps)
        for step_number in range(total_steps):
            observer_count = 0
            observer_distances = [0.0 for i in range(number_agents)]
            summed_distances = 0.0
            temp_reward = 0.0

            # For all agents
            # Calculate distance between poi and agent
            for agent_id in range(number_agents):
                agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[agent_id, step_number, 0]
                agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[agent_id, step_number, 1]
                distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                if distance < min_dist:
                    distance = min_dist
                observer_distances[agent_id] = distance

                # Check if agent observes poi, update closest step distance
                if distance < activation_dist: # Rover is in observation range
                    observer_count += 1

            rearrange_dist_vec(observer_distances)

            # update closest distance only if poi is observed
            if observer_count >= coupling:
                for rv in range(coupling):
                    summed_distances += observer_distances[rv]
                    assert(observer_distances[rv] <= activation_dist)
                temp_reward = poi_values[poi_id]/summed_distances
            else:
                temp_reward = 0.0

            if temp_reward > current_poi_reward:
                current_poi_reward = temp_reward

        g_reward += current_poi_reward

    # CALCULATE DIFFERENCE REWARD
    for agent_id in range(number_agents):
        g_without_self = 0.0

        for poi_id in range(number_pois):
            current_poi_reward = 0.0

            for step_number in range(total_steps):
                # Count how many agents observe poi, update closest distance if necessary
                observer_count = 0
                observer_distances = [0.0 for i in range(number_agents)]
                summed_distances = 0.0
                temp_reward = 0.0

                # Calculate distance between poi and agent
                for other_agent_id in range(number_agents):
                    if agent_id != other_agent_id:
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[other_agent_id, step_number, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[other_agent_id, step_number, 1]
                        distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                        if distance < min_dist:
                            distance = min_dist
                        observer_distances[other_agent_id] = distance
                        
                        # Check if agent observes poi, update closest step distance
                        if distance < activation_dist:
                            observer_count += 1
                    else:
                        observer_distances[agent_id] = inf

                rearrange_dist_vec(observer_distances)
                            
                # update closest distance only if poi is observed    
                if observer_count >= coupling:
                    for rv in range(coupling):
                        summed_distances += observer_distances[rv]
                        assert(observer_distances[rv] <= activation_dist)
                    temp_reward = poi_values[poi_id]/summed_distances
                else:
                    temp_reward = 0.0

                if temp_reward > current_poi_reward:
                    current_poi_reward = temp_reward

            g_without_self += current_poi_reward
        difference_reward[agent_id] = g_reward - g_without_self

    data["Agent Rewards"] = difference_reward
    data["Global Reward"] = g_reward


# D++ REWARD ----------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def calc_dpp_reward(data):
    cdef int number_agents = p.number_of_agents
    cdef int number_pois = p.number_of_pois
    cdef double min_dist = p.min_distance
    cdef int total_steps = p.total_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] agent_pos_history = data["Agent Position History"]
    cdef double[:] poi_values = data['Poi Values']
    cdef double[:, :] poi_positions = data["Poi Positions"]
    cdef int poi_id, step_number, agent_id, observer_count, other_agent_id, counterfactual_count
    cdef double agent_x_dist, agent_y_dist, distance
    cdef double inf = float("inf")
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double g_with_counterfactuals = 0.0 # Reward with n counterfactual partners added
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double temp_dpp_reward = 0.0
    cdef double summed_distances = 0.0
    cdef double[:] dplusplus_reward = np.zeros(number_agents)
    
    # CALCULATE GLOBAL REWARD
    # For all POIs
    for poi_id in range(number_pois):
        current_poi_reward = 0.0

        # For all timesteps (rover steps)
        for step_number in range(total_steps):
            # Count how many agents observe poi, update closest distance if necessary
            observer_count = 0
            observer_distances = [0.0 for i in range(number_agents)]
            summed_distances = 0.0
            temp_reward = 0.0

            # For all agents
            # Calculate distance between poi and agent
            for agent_id in range(number_agents):
                agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[agent_id, step_number, 0]
                agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[agent_id, step_number, 1]
                distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                if distance < min_dist:
                    distance = min_dist
                observer_distances[agent_id] = distance

                # Check if agent observes poi, update closest step distance
                if distance < activation_dist: # Rover is in observation range
                    observer_count += 1

            rearrange_dist_vec(observer_distances)

            # update closest distance only if poi is observed
            if observer_count >= coupling:
                for rv in range(coupling):
                    summed_distances += observer_distances[rv]
                    assert(observer_distances[rv] <= activation_dist)
                temp_reward = poi_values[poi_id]/(0.5*summed_distances)
            else:
                temp_reward = 0.0

            if temp_reward > current_poi_reward:
                current_poi_reward = temp_reward

        g_reward += current_poi_reward

    # CALCULATE DIFFERENCE REWARD
    for agent_id in range(number_agents):
        g_without_self = 0.0

        for poi_id in range(number_pois):
            current_poi_reward = 0.0

            for step_number in range(total_steps):
                # Count how many agents observe poi, update closest distance if necessary
                observer_count = 0
                observer_distances = [0.0 for i in range(number_agents)]
                summed_distances = 0.0
                temp_reward = 0.0

                # Calculate distance between poi and agent
                for other_agent_id in range(number_agents):
                    if agent_id != other_agent_id:
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[other_agent_id, step_number, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[other_agent_id, step_number, 1]
                        distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                        if distance < min_dist:
                            distance = min_dist
                        observer_distances[other_agent_id] = distance

                        # Check if agent observes poi, update closest step distance
                        if distance < activation_dist:
                            observer_count += 1
                    else:
                        observer_distances[other_agent_id] = inf

                rearrange_dist_vec(observer_distances)

                # update closest distance only if poi is observed
                if observer_count >= coupling:
                    for rv in range(coupling):
                        summed_distances += observer_distances[rv]
                        assert(observer_distances[rv] <= activation_dist)
                    temp_reward = poi_values[poi_id]/(0.5*summed_distances)
                else:
                    temp_reward = 0.0

                if temp_reward > current_poi_reward:
                    current_poi_reward = temp_reward

            g_without_self += current_poi_reward

        dplusplus_reward[agent_id] = g_reward - g_without_self
    
    # CALCULATE DPP REWARD
    for counterfactual_count in range(coupling):

        # Calculate Difference with Extra Me Reward
        for agent_id in range(number_agents):
            g_with_counterfactuals = 0.0
            self_dist = 0.0

            for poi_id in range(number_pois):
                current_poi_reward = 0.0

                for step_number in range(total_steps):
                    # Count how many agents observe poi, update closest distance if necessary
                    observer_count = 0
                    observer_distances = [0.0 for i in range(number_agents)]
                    summed_distances = 0.0
                    temp_reward = 0.0

                    # Calculate distance between poi and agent
                    for other_agent_id in range(number_agents):
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[other_agent_id, step_number, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[other_agent_id, step_number, 1]
                        distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                        if distance < min_dist:
                            distance = min_dist
                        observer_distances[other_agent_id] = distance

                        if other_agent_id == agent_id:
                            self_dist = distance # Track distance from self for counterfactuals

                        # Check if agent observes poi, update closest step distance
                        if distance < activation_dist:
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
                            assert(observer_distances[rv] <= activation_dist)
                        temp_reward = poi_values[poi_id]/(0.5*summed_distances)
                    else:
                        temp_reward = 0.0

                    if temp_reward > current_poi_reward:
                        current_poi_reward = temp_reward

                g_with_counterfactuals += current_poi_reward

            temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + counterfactual_count)
            if temp_dpp_reward > dplusplus_reward[agent_id]:
                dplusplus_reward[agent_id] = temp_dpp_reward

    data["Agent Rewards"] = dplusplus_reward
    data["Global Reward"] = g_reward

# S-D++ REWARD --------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def calc_sdpp_reward(data):
    cdef int number_agents = p.number_of_agents
    cdef int number_pois = p.number_of_pois
    cdef double min_dist = p.min_distance
    cdef int total_steps = p.total_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] agent_pos_history = data["Agent Position History"]
    cdef double[:] poi_values = data['Poi Values']
    cdef double[:, :] poi_positions = data["Poi Positions"]
    cdef double[:] rov_partners
    cdef int poi_id, step_number, agent_id, observer_count, other_agent_id, counterfactual_count
    cdef double agent_x_dist, agent_y_dist, distance
    cdef double inf = float("inf")
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double g_with_counterfactuals = 0.0 # Reward with n counterfactual partners added
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double temp_dpp_reward = 0.0
    cdef double summed_distances = 0.0
    cdef double[:] dplusplus_reward = np.zeros(number_agents)

    # CALCULATE GLOBAL REWARD
    # For all POIs
    for poi_id in range(number_pois):
        current_poi_reward = 0.0

        # For all timesteps (rover steps)
        for step_number in range(total_steps):
            # Count how many agents observe poi, update closest distance if necessary
            observer_count = 0
            observer_distances = [0.0 for i in range(number_agents)]
            summed_distances = 0.0
            temp_reward = 0.0

            # For all agents
            # Calculate distance between poi and agent
            for agent_id in range(number_agents):
                agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[agent_id, step_number, 0]
                agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[agent_id, step_number, 1]
                distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                if distance < min_dist:
                    distance = min_dist
                observer_distances[agent_id] = distance

                # Check if agent observes poi, update closest step distance
                if distance <= activation_dist: # Rover is in observation range
                    observer_count += 1

            rearrange_dist_vec(observer_distances)

            # update closest distance only if poi is observed
            if observer_count >= coupling:
                for rv in range(coupling):
                    summed_distances += observer_distances[rv]
                    assert(observer_distances[rv] <= activation_dist)
                temp_reward = poi_values[poi_id]/(0.5*summed_distances)
            else:
                temp_reward = 0.0

            if temp_reward > current_poi_reward:
                current_poi_reward = temp_reward

        g_reward += current_poi_reward

    # CALCULATE DIFFERENCE REWARD
    for agent_id in range(number_agents):
        g_without_self = 0.0

        for poi_id in range(number_pois):
            current_poi_reward = 0.0

            for step_number in range(total_steps):
                # Count how many agents observe poi, update closest distance if necessary
                observer_count = 0
                observer_distances = [0.0 for i in range(number_agents)]
                summed_distances = 0.0
                temp_reward = 0.0

                # Calculate distance between poi and agent
                for other_agent_id in range(number_agents):
                    if agent_id != other_agent_id:
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[other_agent_id, step_number, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[other_agent_id, step_number, 1]
                        distance = math.sqrt((agent_x_dist * agent_x_dist) + (agent_y_dist * agent_y_dist))
                        if distance < min_dist:
                            distance = min_dist
                        observer_distances[other_agent_id] = distance

                        # Check if agent observes poi, update closest step distance
                        if distance <= activation_dist:
                            observer_count += 1
                    else:
                        observer_distances[other_agent_id] = inf

                rearrange_dist_vec(observer_distances)

                # update closest distance only if poi is observed
                if observer_count >= coupling:
                    for rv in range(coupling):
                        summed_distances += observer_distances[rv]
                        assert(observer_distances[rv] <= activation_dist)
                    temp_reward = poi_values[poi_id]/(0.5*summed_distances)
                else:
                    temp_reward = 0.0

                if temp_reward > current_poi_reward:
                    current_poi_reward = temp_reward

            g_without_self += current_poi_reward

        dplusplus_reward[agent_id] = g_reward - g_without_self

    # CALCULATE S-DPP REWARD
    for counterfactual_count in range(coupling):

        # Calculate reward with suggested counterfacual partners
        for agent_id in range(number_agents):
            g_with_counterfactuals = 0.0
            self_dist = 0.0

            for poi_id in range(number_pois):
                current_poi_reward = 0.0

                for step_number in range(total_steps):
                    # Count how many agents observe poi, update closest distance if necessary
                    observer_count = 0
                    observer_distances = [0.0 for i in range(number_agents)]
                    summed_distances = 0.0
                    temp_reward = 0.0

                    # Calculate distance between poi and agent
                    for other_agent_id in range(number_agents):
                        agent_x_dist = poi_positions[poi_id, 0] - agent_pos_history[other_agent_id, step_number, 0]
                        agent_y_dist = poi_positions[poi_id, 1] - agent_pos_history[other_agent_id, step_number, 1]
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
                        rov_partners = closest_others(data, step_number, counterfactual_count, agent_id, poi_id)
                        # rov_partners = random_partners(data, step_number, counterfactual_count, agent_id, poi_id)
                        # rov_partners = value_based_partners(data, counterfactual_count, poi_id)

                        for rovid in range(counterfactual_count):
                            observer_distances.append(rov_partners[rovid])  # Append n closest
                        observer_count += counterfactual_count

                    rearrange_dist_vec(observer_distances)  # Rearrange rover distances with added counterfactuals

                    # update closest distance only if poi is observed
                    if observer_count >= coupling:
                        for rv in range(coupling):
                            summed_distances += observer_distances[rv]
                            # assert(observer_distances[rv] <= activation_dist)
                        temp_reward = poi_values[poi_id]/(0.5*summed_distances)
                    else:
                        temp_reward = 0.0

                    if temp_reward > current_poi_reward:
                        current_poi_reward = temp_reward

                g_with_counterfactuals += current_poi_reward

            temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + counterfactual_count)
            if temp_dpp_reward > dplusplus_reward[agent_id]:
                dplusplus_reward[agent_id] = temp_dpp_reward

    data["Agent Rewards"] = dplusplus_reward
    data["Global Reward"] = g_reward
