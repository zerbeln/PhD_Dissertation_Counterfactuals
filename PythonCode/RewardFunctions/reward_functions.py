import numpy as np
from parameters import parameters as p


# DIFFERENCE REWARDS --------------------------------------------------------------------------------------------------
def calc_difference(pois, global_reward):
    """
    Calculate each rover's difference reward at the current time step
    """
    obs_rad = p["observation_radius"]
    n_rovers = p["n_rovers"]

    difference_rewards = np.zeros(n_rovers)
    for agent_id in range(n_rovers):  # For each rover
        counterfactual_global_reward = 0.0
        for pk in pois:  # For each POI
            observer_count = 0
            rover_distances = pois[pk].observer_distances.copy()
            rover_distances[agent_id] = 1000.00
            rover_distances = np.sort(rover_distances)

            for i in range(int(pois[pk].coupling)):
                dist = rover_distances[i]
                if dist < obs_rad:
                    observer_count += 1

            # Compute reward if coupling is satisfied
            if observer_count >= int(pois[pk].coupling):
                summed_observer_distances = sum(rover_distances[0:int(pois[pk].coupling)])
                counterfactual_global_reward += pois[pk].value/(summed_observer_distances/pois[pk].coupling)

        difference_rewards[agent_id] = global_reward - counterfactual_global_reward

    return difference_rewards


# D++ REWARD ----------------------------------------------------------------------------------------------------------
def calc_dpp(pois, global_reward):
    """
    Calculate D++ rewards for each rover at the current time step
    """
    n_rovers = p["n_rovers"]
    obs_rad = p["observation_radius"]
    d_rewards = calc_difference(pois, global_reward)
    dpp_rewards = np.zeros(n_rovers)

    # Calculate D++ Reward with (TotalAgents - 1) Counterfactuals
    n_counters = p["coupling"] - 1
    for agent_id in range(n_rovers):
        counterfactual_global_reward = 0.0
        for pk in pois:
            observer_count = 0

            rover_distances = pois[pk].observer_distances.copy()
            counterfactual_rovers = np.ones(n_counters) * pois[pk].observer_distances[agent_id]
            rover_distances = np.append(rover_distances, counterfactual_rovers)
            rover_distances = np.sort(rover_distances)

            for i in range(int(pois[pk].coupling)):
                if rover_distances[i] < obs_rad:
                    observer_count += 1

            # Update POI observers
            if observer_count >= pois[pk].coupling:
                summed_observer_distances = sum(rover_distances[0:int(pois[pk].coupling)])
                counterfactual_global_reward += pois[pk].value/(summed_observer_distances/pois[pk].coupling)

        dpp_rewards[agent_id] = (counterfactual_global_reward - global_reward) / n_counters

    for agent_id in range(n_rovers):
        if dpp_rewards[agent_id] > d_rewards[agent_id]:
            dpp_rewards[agent_id] = 0.0

            for c in range(p["coupling"]-1):
                n_counters = c + 1
                counterfactual_global_reward = 0.0
                for pk in pois:
                    observer_count = 0
                    rover_distances = pois[pk].observer_distances.copy()
                    counterfactual_rovers = np.ones(n_counters) * pois[pk].observer_distances[agent_id]
                    rover_distances = np.append(rover_distances, counterfactual_rovers)
                    rover_distances = np.sort(rover_distances)

                    for i in range(int(pois[pk].coupling)):
                        if rover_distances[i] < obs_rad:
                            observer_count += 1

                    # Update POI observers
                    if observer_count >= pois[pk].coupling:
                        summed_observer_distances = sum(rover_distances[0:int(pois[pk].coupling)])
                        counterfactual_global_reward += pois[pk].value / (summed_observer_distances/pois[pk].coupling)

                # Calculate D++ reward with n counterfactuals added
                temp_dpp = (counterfactual_global_reward - global_reward)/n_counters
                if temp_dpp > dpp_rewards[agent_id]:
                    dpp_rewards[agent_id] = temp_dpp
                    c = int(p["coupling"] + 1)  # Stop iterrating
        else:
            dpp_rewards[agent_id] = d_rewards[agent_id]  # Returns difference reward

    return dpp_rewards

