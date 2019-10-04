import numpy as np
import math
import sys

def low_high_split(rover_dist, rover_id, poi_id, poi_values, n_counters):
    """
    Rovers with even IDs go for high value POIs, Rovers with odd IDs go for low value POIs
    :param rover_dist:
    :param rover_id:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :return: partners
    """
    partners = np.zeros(n_counters)

    if rover_id % 2 == 0:
        if poi_values[poi_id] > 5.0:
            for partner_id in range(n_counters):
                partners[partner_id] = rover_dist
        else:
            for partner_id in range(n_counters):
                partners[partner_id] = 100.0
    else:
        if poi_values[poi_id] <= 5.0:
            for partner_id in range(n_counters):
                partners[partner_id] = rover_dist
        else:
            for partner_id in range(n_counters):
                partners[partner_id] = 100.0

    return partners

def high_value_only(rover_dist, poi_id, poi_values, n_counters):
    """
    Suggestions give rover stepping stone reward only for high value POIs
    :param rover_dist:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :return: partners
    """
    partners = np.zeros(n_counters)

    if poi_values[poi_id] > 5.0:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.0

    return partners

def low_value_only(rover_dist, poi_id, poi_values, n_counters):
    """
    Suggestions give rover stepping stone reward only for low value POIs
    :param rover_dist:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :return: partners
    """
    partners = np.zeros(n_counters)

    if poi_values[poi_id] <= 5.0:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.0

    return partners


def value_based_incentives(p, rover_dist, poi_id, poi_values, n_counters):
    """
    Partners are placed close to high value POIs to generate larger stepping stone reward
    Partners are placed further away from low value POIs to generate smaller stepping stone reward
    :param rover_dist:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :return: partners
    """
    partners = np.zeros(n_counters)

    if rover_dist < p.min_observation_dist:
        if poi_values[poi_id] > 5:
            for partner_id in range(n_counters):
                partners[partner_id] = p.min_distance
        else:
            for partner_id in range(n_counters):
                partners[partner_id] = p.min_observation_dist - 0.01
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.0

    return partners

def partner_proximity_suggestions(p, rover_dist, n_counters, self_id, rover_paths, step_id):
    """
    Partner suggestions based on rover proximity to other rovers
    :param rover_dist:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :param rover_paths:
    :param step_id:
    :return: partners
    """
    partners = np.zeros(n_counters)
    count = 0

    # Calculate distance between self and other rovers
    self_x = rover_paths[step_id, self_id, 0]
    self_y = rover_paths[step_id, self_id, 1]
    for rover_id in range(p.num_rovers):
        if rover_id == self_id:
            continue
        rover_x = rover_paths[step_id, rover_id, 0]
        rover_y = rover_paths[step_id, rover_id, 1]
        x_dist = rover_x - self_x
        y_dist = rover_y - self_y
        dist = math.sqrt((x_dist**2)+(y_dist**2))
        if dist < p.min_observation_dist:
            count += 1

    if count > 0:
        if rover_dist < p.min_observation_dist:
            for partner_id in range(n_counters):
                partners[partner_id] = p.min_distance
        else:
            for partner_id in range(n_counters):
                partners[partner_id] = p.min_observation_dist - 0.01
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = p.min_distance

    return partners

def go_left_suggestions(p, rover_dist, poi_id, poi_pos, n_counters):
    partners = np.zeros(n_counters)
    x_middle = p.x_dim/2

    if poi_pos[poi_id, 0] < x_middle:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.00

    return partners

def go_right_suggestions(p, rover_dist, poi_id, poi_pos, n_counters):
    partners = np.zeros(n_counters)
    x_middle = p.x_dim / 2

    if poi_pos[poi_id, 0] > x_middle:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.00

    return partners

def left_right_split(p, rover_dist, rover_id, poi_id, poi_pos, n_counters):
    partners = np.zeros(n_counters)
    x_middle = p.x_dim / 2

    if rover_id % 2 == 0 and poi_pos[poi_id, 0] > x_middle:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    elif rover_id % 2 != 0 and poi_pos[poi_id, 0] < x_middle:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.00

    return partners

def get_counterfactual_partners(p, n_counters, self_id, rover_dist, rover_paths, poi_id, poi_values, poi_pos, step_id, suggestion):
    partners = np.zeros(n_counters)

    if suggestion == "none":
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    elif suggestion == "high_val":
        partners = high_value_only(rover_dist, poi_id, poi_values, n_counters)
    elif suggestion == "low_val":
        partners = low_value_only(rover_dist, poi_id, poi_values, n_counters)
    elif suggestion == "high_low":
        partners = low_high_split(rover_dist, self_id, poi_id, poi_values, n_counters)
    elif suggestion == "value_incentives":
        partners = value_based_incentives(p, rover_dist, poi_id, poi_values, n_counters)
    elif suggestion == "partner_proximity":
        partners = partner_proximity_suggestions(p, rover_dist, n_counters, self_id, rover_paths, step_id)
    elif suggestion == "left":
        partners = go_left_suggestions(p, rover_dist, poi_id, poi_pos, n_counters)
    elif suggestion == "right":
        partners = go_right_suggestions(p, rover_dist, poi_id, poi_pos, n_counters)
    elif suggestion == "left_right":
        partners = left_right_split(p, rover_dist, self_id, poi_id, poi_pos, n_counters)
    else:
        sys.exit('Incorrect Suggestion Type')

    return partners
