import numpy as np
from parameters import Parameters as p
import math

# GLOBAL REWARDS ------------------------------------------------------------------------------------------------------
def calc_global(rover_path, poi_values, poi_positions):
    num_steps = p.num_steps + 1
    inf = 1000.00
    g_reward = 0.0

    # For all POIs
    for poi_id in range(p.num_pois):
        current_poi_reward = 0.0

        # For all timesteps (rover steps)
        for step_number in range(num_steps):
            observer_count = 0  # Track number of observers for given POI
            observer_distances = [0.0 for i in range(p.num_rovers)]
            summed_distances = 0.0  # Denominator of reward function
            temp_reward = 0.0  # Tracks reward given by POI for each time step

            # For all agents
            # Calculate distance between poi and agent
            for rover_id in range(p.num_rovers):
                rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, rover_id, 0]
                rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, rover_id, 1]
                distance = math.sqrt((rover_x_dist * rover_x_dist) + (rover_y_dist * rover_y_dist))
                if distance <= p.min_distance:
                    distance = p.min_distance  # Clip distance
                observer_distances[rover_id] = distance

                # Check if agent observes poi
                if distance <= p.activation_dist: # Rover is in observation range
                    observer_count += 1

            if observer_count >= p.coupling:  # If observers meet coupling req, calculate reward
                for rv in range(p.coupling):
                    summed_distances += min(observer_distances)
                    od_index = observer_distances.index(min(observer_distances))
                    observer_distances[od_index] = inf
                temp_reward = poi_values[poi_id]/summed_distances
            else:
                temp_reward = 0.0

            if temp_reward > current_poi_reward:
                current_poi_reward = temp_reward

        g_reward += current_poi_reward

    return g_reward


# DIFFERENCE REWARDS -------------------------------------------------------------------------------------------------
def calc_difference(rover_path, poi_values, poi_positions):
    num_steps = p.num_steps + 1
    inf = 1000.00
    difference_rewards = np.zeros(p.num_rovers * p.num_types)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global(rover_path, poi_values, poi_positions)

    # CALCULATE DIFFERENCE REWARD
    for agent_id in range(p.num_rovers):
        g_without_self = 0.0

        for poi_id in range(p.num_pois):
            current_poi_reward = 0.0

            for step_number in range(num_steps):
                observer_count = 0  # Track number of POI observers at time step
                observer_distances = [0.0 for i in range(p.num_rovers)]
                summed_distances = 0.0  # Denominator of reward function
                temp_reward = 0.0  # Tracks reward given by POI for each time step

                # Calculate distance between poi and agent
                for other_agent_id in range(p.num_rovers):
                    if agent_id != other_agent_id:
                        rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, other_agent_id, 0]
                        rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, other_agent_id, 1]
                        distance = math.sqrt((rover_x_dist * rover_x_dist) + (rover_y_dist * rover_y_dist))
                        if distance <= p.min_distance:
                            distance = p.min_distance
                        observer_distances[other_agent_id] = distance

                        # Check if agent observes poi, update closest step distance
                        if distance < p.activation_dist:
                            observer_count += 1
                    else:
                        observer_distances[agent_id] = inf  # Ignore self

                if observer_count >= p.coupling: # If coupling satisfied, compute reward
                    for rv in range(p.coupling):
                        summed_distances += min(observer_distances)
                        od_index = observer_distances.index(min(observer_distances))
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
def calc_dpp(rover_path, poi_values, poi_positions):
    num_steps = p.num_steps + 1
    inf = 1000.00
    dplusplus_reward = np.zeros(p.num_rovers * p.num_types)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global(rover_path, poi_values, poi_positions)

    # CALCULATE DIFFERENCE REWARD
    difference_reward = calc_difference(rover_path, poi_values, poi_positions)

    # CALCULATE DPP REWARD
    for counterfactual_count in range(p.coupling):

        # Calculate Difference with Extra Me Reward
        for agent_id in range(p.num_rovers):
            g_with_counterfactuals = 0.0
            self_dist = 0.0

            for poi_id in range(p.num_pois):
                current_poi_reward = 0.0

                for step_number in range(num_steps):
                    observer_count = 0  # Track number of POI observers at time step
                    observer_distances = []
                    summed_distances = 0.0 # Denominator of reward function
                    temp_reward = 0.0  # Tracks reward given by POI for each time step

                    # Calculate distance between poi and agent
                    for other_agent_id in range(p.num_rovers):
                        rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, other_agent_id, 0]
                        rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, other_agent_id, 1]
                        distance = math.sqrt((rover_x_dist * rover_x_dist) + (rover_y_dist * rover_y_dist))
                        if distance < p.min_distance:
                            distance = p.min_distance
                        observer_distances.append(distance)

                        if other_agent_id == agent_id:
                            self_dist = distance # Track distance from self for counterfactuals

                        # Update observer count
                        if distance <= p.activation_dist:
                            observer_count += 1

                    if observer_count < p.coupling:  # Add counterfactual partners if needed
                        if self_dist <= p.activation_dist:
                            for c in range(counterfactual_count):
                                observer_distances.append(self_dist)
                            observer_count += counterfactual_count

                    if observer_count >= p.coupling:  # If coupling satisfied, compute reward
                        for rv in range(p.coupling):
                            summed_distances += min(observer_distances)
                            od_index = observer_distances.index(min(observer_distances))
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

    for rov_id in range(p.num_rovers):
        if difference_reward[rov_id] > dplusplus_reward[rov_id]:  # Use difference reward, if it is better
            dplusplus_reward[rov_id] = difference_reward[rov_id]

    return dplusplus_reward
