import numpy as np
import math
from AADI_RoverDomain.parameters import Parameters as p

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

    npartners = n_counters
    partners = np.zeros(npartners)

    if rover_id % 2 == 0:
        if poi_values[poi_id] > 5.0:
            for partner_id in range(npartners):
                partners[partner_id] = rover_dist
        else:
            for partner_id in range(npartners):
                partners[partner_id] = 100.0
    else:
        if poi_values[poi_id] <= 5.0:
            for partner_id in range(npartners):
                partners[partner_id] = rover_dist
        else:
            for partner_id in range(npartners):
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

    npartners = n_counters
    partners = np.zeros(npartners)

    if poi_values[poi_id] > 5.0:
        for partner_id in range(npartners):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(npartners):
            partners[partner_id] = 100.0

    return partners

def low_value_pois(rover_dist, poi_id, poi_values, n_counters):
    """
    Suggestions give rover stepping stone reward only for low value POIs
    :param rover_dist:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :return: partners
    """

    npartners = n_counters
    partners = np.zeros(npartners)

    if poi_values[poi_id] <= 5.0:
        for partner_id in range(npartners):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(npartners):
            partners[partner_id] = 100.0

    return partners


def value_based_suggestions(rover_dist, poi_id, poi_values, n_counters):
    """
    Partners are placed close to high value POIs to generate larger stepping stone reward
    Partners are placed further away from low value POIs to generate smaller stepping stone reward
    :param rover_dist:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :return: partners
    """

    npartners = n_counters
    partners = np.zeros(npartners)

    if rover_dist < p.min_observation_dist:
        if poi_values[poi_id] > 5:
            for partner_id in range(npartners):
                partners[partner_id] = p.min_distance
        else:
            for partner_id in range(npartners):
                partners[partner_id] = p.min_observation_dist - 0.01
    else:
        for partner_id in range(npartners):
            partners[partner_id] = 100.0

    return partners

def team_member_based_suggestions(rover_dist, n_counters, self_id, rover_paths, step_id):
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

    npartners = n_counters
    partners = np.zeros(npartners)
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
            for partner_id in range(npartners):
                partners[partner_id] = p.min_distance
        else:
            for partner_id in range(npartners):
                partners[partner_id] = p.min_observation_dist - 0.01
    else:
        for partner_id in range(npartners):
            partners[partner_id] = p.min_distance

    return partners
