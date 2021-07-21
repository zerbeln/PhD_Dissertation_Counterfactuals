import numpy as np
import math
from Python_Code.supervisor import get_counterfactual_partners, get_counterfactual_action
from Python_Code.reward_functions import calc_difference_tight
from parameters import parameters as p


# S-Difference REWARD -------------------------------------------------------------------------------------------------
def calc_sd_reward(rover_paths, poi, global_reward, sgst):
    """
    Calcualte each rover's difference reward with suggestions from entire rover trajectory
    :param rover_paths:  X-Y coordinates of each rover at each time step
    :param poi: np array with X-Y coordinates and value for each POI
    :param global_reward:  Reward given to the team from the world
    :return: difference_rewards (np array of size (n_rovers))
    """
    total_steps = int(p["n_steps"] + 1)  # The +1 is to account for the initial position
    inf = 1000.00

    difference_rewards = np.zeros(p["n_rovers"])

    for agent_id in range(p["n_rovers"]):  # For each rover
        poi_observer_distances = np.zeros((p["n_poi"], total_steps))  # Tracks summed observer distances
        poi_observed = np.zeros(p["n_poi"])

        for poi_id in range(p["n_poi"]):  # For each POI
            for step_index in range(total_steps):  # For each step in trajectory
                observer_count = 0
                rover_distances = np.zeros(p["n_rovers"])  # Track distances between rovers and POI

                # Count how many agents observe poi, update closest distances
                for other_agent_id in range(p["n_rovers"]):
                    if agent_id != other_agent_id:  # Remove current rover's trajectory
                        # Calculate separation distance between poi and agent
                        x_distance = poi[poi_id, 0] - rover_paths[other_agent_id, step_index, 0]
                        y_distance = poi[poi_id, 1] - rover_paths[other_agent_id, step_index, 1]
                        distance = math.sqrt((x_distance**2) + (y_distance**2))

                        if distance < p["min_distance"]:
                            distance = p["min_distance"]

                        rover_distances[other_agent_id] = distance

                        # Check if agent observes poi
                        if distance < p["obs_rad"]:
                            observer_count += 1
                    else:
                        x_distance = poi[poi_id, 0] - rover_paths[agent_id, step_index, 0]
                        y_distance = poi[poi_id, 1] - rover_paths[agent_id, step_index, 1]
                        distance = math.sqrt((x_distance**2) + (y_distance**2))

                        if distance <= p["obs_rad"]:
                            rover_distances[agent_id] = get_counterfactual_action(distance, agent_id, poi_id, poi, sgst)
                        else:
                            rover_distances[agent_id] = inf

                        if rover_distances[agent_id] < p["obs_rad"]:
                            observer_count += 1

                # Determine if coupling is satisfied
                if observer_count >= p["coupling"]:
                    summed_observer_distances = 0.0
                    poi_observed[poi_id] = 1
                    for observer in range(p["coupling"]):  # Sum distances of closest observers
                        summed_observer_distances += min(rover_distances)
                        od_index = np.argmin(rover_distances)
                        rover_distances[od_index] = inf
                    poi_observer_distances[poi_id, step_index] = summed_observer_distances
                else:
                    poi_observer_distances[poi_id, step_index] = inf

        counterfactual_global_reward = 0.0
        for poi_id in range(p["n_poi"]):
            if poi_observed[poi_id] == 1:
                counterfactual_global_reward += poi[poi_id, 2] / (min(poi_observer_distances[poi_id])/p["coupling"])
        difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards


# S-D++ REWARD -------------------------------------------------------------------------------------------------------
def calc_sdpp(rover_paths, poi, global_reward, sgst):
    """
    Calculate S-D++ rewards for each rover across entire trajectory
    :param rover_paths: X-Y coordinates of each rover at each time step
    :param poi: np array with X-Y coordinates and value for each POI
    :param global_reward: Reward given to the team from the world
    :param sgst: String indicating which type of suggestion to use
    :return: dpp_rewards (np array of size (n_rovers))
    """

    total_steps = int(p["n_steps"] + 1)  # The +1 is to account for the initial position (step 0)
    inf = 1000.00
    dpp_rewards = np.zeros(p["n_rovers"])
    difference_rewards = calc_difference_tight(rover_paths, poi, global_reward)

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    n_counters = p["coupling"] - 1

    for agent_id in range(p["n_rovers"]):
        poi_observer_distances = np.zeros((p["n_poi"], total_steps))
        poi_observed = np.zeros(p["n_poi"])

        for poi_id in range(p["n_poi"]):
            for step_index in range(total_steps):
                observer_count = 0
                rover_distances = np.zeros(p["n_rovers"] + n_counters)

                # Calculate linear distances between POI and agents, count observers
                for other_agent_id in range(p["n_rovers"]):
                    x_distance = poi[poi_id, 0] - rover_paths[other_agent_id, step_index, 0]
                    y_distance = poi[poi_id, 1] - rover_paths[other_agent_id, step_index, 1]
                    distance = math.sqrt((x_distance**2) + (y_distance**2))

                    if distance < p["min_distance"]:
                        distance = p["min_distance"]

                    rover_distances[other_agent_id] = distance

                    if distance < p["obs_rad"]:
                        observer_count += 1

                # Get suggestion from supervisor if rover has discovered a POI
                if rover_distances[agent_id] <= p["obs_rad"]:
                    counterfactual_agents = get_counterfactual_partners(n_counters, agent_id, rover_distances[agent_id], poi_id, poi, sgst)
                    for partner_id in range(n_counters):
                        rover_distances[p["n_rovers"] + partner_id] = counterfactual_agents[partner_id]

                        if abs(counterfactual_agents[partner_id]) < p["obs_rad"]:
                            observer_count += 1

                # Update POI observers
                if observer_count >= p["coupling"]:
                    summed_observer_distances = 0.0
                    poi_observed[poi_id] = 1
                    for observer in range(p["coupling"]):  # Sum distances of closest observers
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
        for poi_id in range(p["n_poi"]):
            if poi_observed[poi_id] == 1:
                counterfactual_global_reward += poi[poi_id, 2]/(min(poi_observer_distances[poi_id])/p["coupling"])
        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / n_counters

    for agent_id in range(p["n_rovers"]):
        if abs(dpp_rewards[agent_id]) > difference_rewards[agent_id]:
            dpp_rewards[agent_id] = 0.0
            poi_observer_distances = np.zeros((p["n_poi"], total_steps))
            poi_observed = np.zeros(p["n_poi"])

            for n_counters in range(p["coupling"]):
                if n_counters == 0:  # 0 counterfactual partnrs is identical to G
                    n_counters = 1
                for poi_id in range(p["n_poi"]):
                    for step_index in range(total_steps):
                        observer_count = 0
                        rover_distances = np.zeros(p["n_rovers"] + n_counters)

                        # Calculate linear distances between POI and agents, count observers
                        for other_agent_id in range(p["n_rovers"]):
                            x_distance = poi[poi_id, 0] - rover_paths[other_agent_id, step_index, 0]
                            y_distance = poi[poi_id, 1] - rover_paths[other_agent_id, step_index, 1]
                            distance = math.sqrt((x_distance**2) + (y_distance**2))

                            if distance < p["min_distance"]:
                                distance = p["min_distance"]

                            rover_distances[other_agent_id] = distance

                            if distance < p["obs_rad"]:
                                observer_count += 1

                        # Get suggestion from supervisor if rover has discovered a POI
                        if rover_distances[agent_id] <= p["obs_rad"]:
                            counterfactual_agents = get_counterfactual_partners(n_counters, agent_id, rover_distances[agent_id], poi_id, poi, sgst)
                            for partner_id in range(n_counters):
                                rover_distances[p["n_rovers"] + partner_id] = counterfactual_agents[partner_id]

                                if abs(counterfactual_agents[partner_id]) < p["obs_rad"]:
                                    observer_count += 1

                        # Update POI observers
                        if observer_count >= p["coupling"]:
                            summed_observer_distances = 0.0
                            poi_observed[poi_id] = 1
                            for observer in range(p["coupling"]):  # Sum distances of closest observers
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
                for poi_id in range(p["n_poi"]):
                    if poi_observed[poi_id] == 1:
                        counterfactual_global_reward += poi[poi_id, 2]/(min(poi_observer_distances[poi_id])/p["coupling"])
                dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward)/n_counters
                if dpp_rewards[agent_id] > difference_rewards[agent_id]:
                    n_counters = p["coupling"] + 1  # Stop iterrating
        else:
            dpp_rewards[agent_id] = difference_rewards[agent_id]  # Returns difference reward

    return dpp_rewards

def sdpp_and_sd(rover_paths, poi, global_reward, sgst):
    """
    Calculate S-D++ rewards for each rover across entire trajectory
    :param rover_paths: X-Y coordinates of each rover at each time step
    :param poi: np array with X-Y coordinates and value for each POI
    :param global_reward: Reward given to the team from the world
    :param sgst: String indicating which type of suggestion to use
    :return: dpp_rewards (np array of size (n_rovers))
    """
    total_steps = int(p["n_steps"] + 1)  # The +1 is to account for the initial position (step 0)
    inf = 1000.00
    difference_rewards = calc_sd_reward(rover_paths, poi, global_reward, sgst, )
    dpp_rewards = np.zeros(p["n_rovers"])

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    n_counters = p["coupling"] - 1
    for agent_id in range(p["n_rovers"]):
        poi_observer_distances = np.zeros((p["n_poi"], total_steps))
        poi_observed = np.zeros(p["n_poi"])

        for poi_id in range(p["n_poi"]):
            for step_index in range(total_steps):
                observer_count = 0
                rover_distances = np.zeros(p["n_rovers"] + n_counters)

                # Calculate linear distances between POI and agents, count observers
                for other_agent_id in range(p["n_rovers"]):
                    x_distance = poi[poi_id, 0] - rover_paths[other_agent_id, step_index, 0]
                    y_distance = poi[poi_id, 1] - rover_paths[other_agent_id, step_index, 1]
                    distance = math.sqrt((x_distance**2) + (y_distance**2))

                    if distance < p["min_distance"]:
                        distance = p["min_distance"]

                    rover_distances[other_agent_id] = distance

                    if distance < p["obs_rad"]:
                        observer_count += 1

                # Get suggestion from supervisor if rover has discovered a POI
                if rover_distances[agent_id] <= p["obs_rad"]:
                    counterfactual_agents = get_counterfactual_partners(n_counters, agent_id, rover_distances[agent_id], poi_id, poi, sgst)
                    for partner_id in range(n_counters):
                        rover_distances[p["n_rovers"] + partner_id] = counterfactual_agents[partner_id]

                        if abs(counterfactual_agents[partner_id]) < p["obs_rad"]:
                            observer_count += 1

                # Update POI observers
                if observer_count >= p["coupling"]:
                    summed_observer_distances = 0.0
                    poi_observed[poi_id] = 1
                    for observer in range(p["coupling"]):  # Sum distances of closest observers
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
        for poi_id in range(p["n_poi"]):
            if poi_observed[poi_id] == 1:
                counterfactual_global_reward += poi[poi_id, 2]/(min(poi_observer_distances[poi_id])/p["coupling"])
        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / n_counters

    for agent_id in range(p["n_rovers"]):
        if abs(dpp_rewards[agent_id]) > difference_rewards[agent_id]:
            dpp_rewards[agent_id] = 0.0
            poi_observer_distances = np.zeros((p["n_poi"], total_steps))
            poi_observed = np.zeros(p["n_poi"])

            for n_counters in range(p["coupling"]):
                if n_counters == 0:  # 0 counterfactual partnrs is identical to G
                    n_counters = 1
                for poi_id in range(p["n_poi"]):
                    suggestion = sgst
                    for step_index in range(total_steps):
                        observer_count = 0
                        rover_distances = np.zeros(p["n_rovers"] + n_counters)

                        # Calculate linear distances between POI and agents, count observers
                        for other_agent_id in range(p["n_rovers"]):
                            x_distance = poi[poi_id, 0] - rover_paths[other_agent_id, step_index, 0]
                            y_distance = poi[poi_id, 1] - rover_paths[other_agent_id, step_index, 1]
                            distance = math.sqrt((x_distance**2) + (y_distance**2))

                            if distance < p["min_distance"]:
                                distance = p["min_distance"]

                            rover_distances[other_agent_id] = distance

                            if distance < p["obs_rad"]:
                                observer_count += 1

                        # Get suggestion from supervisor if rover has discovered a POI
                        if rover_distances[agent_id] <= p["obs_rad"]:
                            counterfactual_agents = get_counterfactual_partners(n_counters, agent_id, rover_distances[agent_id], poi_id, poi, sgst)
                            for partner_id in range(n_counters):
                                rover_distances[p["n_rovers"] + partner_id] = counterfactual_agents[partner_id]

                                if abs(counterfactual_agents[partner_id]) < p["obs_rad"]:
                                    observer_count += 1

                        # Update POI observers
                        if observer_count >= p["coupling"]:
                            summed_observer_distances = 0.0
                            poi_observed[poi_id] = 1
                            for observer in range(p["coupling"]):  # Sum distances of closest observers
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
                for poi_id in range(p["n_poi"]):
                    if poi_observed[poi_id] == 1:
                        counterfactual_global_reward += poi[poi_id, 2]/(min(poi_observer_distances[poi_id])/p["coupling"])
                dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward)/n_counters
                if dpp_rewards[agent_id] > difference_rewards[agent_id]:
                    n_counters = p["coupling"] + 1  # Stop iterrating
        else:
            dpp_rewards[agent_id] = difference_rewards[agent_id]  # Returns difference reward

    return dpp_rewards

