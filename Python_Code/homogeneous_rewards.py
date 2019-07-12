import numpy as np
import math
from AADI_RoverDomain.parameters import Parameters as p


# DIFFERENCE REWARDS --------------------------------------------------------------------------------------------------
def calc_difference(rover_paths, poi_values, poi_positions, global_reward):
    number_agents = p.num_rovers
    number_pois = p.num_pois
    total_steps = p.num_steps + 1
    min_obs_distance = p.min_observation_dist
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


                # Update reward if coupling is satisfied
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
    min_obs_distance = p.min_observation_dist
    inf = 1000.00

    difference_rewards = calc_difference(rover_paths, poi_values, poi_positions, global_reward)
    dpp_rewards = [0.0 for i in range(number_agents)]

    # Calculate Dpp Reward with (TotalAgents - 1) Counterfactuals
    n_counters = p.coupling
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
                    counterfactual_global_reward += poi_values[poi_id] / ((1 / p.coupling) * summed_observer_distances)

            temp_dpp_reward = (counterfactual_global_reward - global_reward) / (1 + n_counters)

            if dpp_rewards[agent_id] < temp_dpp_reward:
                dpp_rewards[agent_id] = temp_dpp_reward

    for agent_id in range(number_agents):

        if dpp_rewards[agent_id] > difference_rewards[agent_id]:
            for n_counters in range(p.coupling):

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
                            if observer_distances[number_agents + partner_id] < min_obs_distance:
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
