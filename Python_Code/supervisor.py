import numpy as np
import sys
from parameters import parameters as p


# COUNTERFACTUAL PARTNER SUGGESTIONS ---------------------------------------------------------------------------------
def low_high_split(rover_dist, rover_id, poi_id, pois, n_counters):
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

    if rover_id % 2 == 0:  # Even IDed rovers pursue higher value targets
        if pois[poi_id, 2] > 5.0:
            for partner_id in range(n_counters):
                partners[partner_id] = rover_dist
        else:
            for partner_id in range(n_counters):
                partners[partner_id] = 100.00
    else:
        if pois[poi_id, 2] <= 5.0:  # Odd IDed rovers pursue lower value targets
            for partner_id in range(n_counters):
                partners[partner_id] = rover_dist
        else:
            for partner_id in range(n_counters):
                partners[partner_id] = 100.00

    return partners


def high_value_only(rover_dist, poi_id, pois, n_counters):
    """
    Suggestions give rover stepping stone reward only for high value POIs
    :param rover_dist:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :return: partners
    """
    partners = np.zeros(n_counters)

    if pois[poi_id, 2] > 5.0:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.00

    return partners


def low_value_only(rover_dist, poi_id, pois, n_counters):
    """
    Suggestions give rover stepping stone reward only for low value POIs
    :param rover_dist:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :return: partners
    """
    partners = np.zeros(n_counters)

    if pois[poi_id, 2] <= 5.0:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.00

    return partners


def value_based_incentives(rover_dist, poi_id, pois, n_counters):
    """
    :param rover_dist:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :param min_dist:
    :param obs_rad:
    :return:
    """
    partners = np.zeros(n_counters)

    if rover_dist < p["min_obs_rad"]:
        if pois[poi_id, 2] > 5:
            for partner_id in range(n_counters):
                partners[partner_id] = p["min_dist"]
        else:
            for partner_id in range(n_counters):
                partners[partner_id] = p["min_obs_rad"] - 0.01
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.0

    return partners


def get_counterfactual_partners(n_counters, self_id, rover_dist, poi_id, pois, suggestion):
    partners = np.zeros(n_counters)

    if suggestion == "high_val":
        partners = high_value_only(rover_dist, poi_id, pois, n_counters)
    elif suggestion == "low_val":
        partners = low_value_only(rover_dist, poi_id, pois, n_counters)
    elif suggestion == "high_low":
        partners = low_high_split(rover_dist, self_id, poi_id, pois, n_counters)
    elif suggestion == "val_based":
        partners = value_based_incentives(rover_dist, poi_id, pois, n_counters)
    else:
        sys.exit('Incorrect Suggestion Type')

    return partners


# COUNTERFACTUAL ACTION SUGGESTIONS -----------------------------------------------------------------------------------
def high_val_action_suggestions(rover_dist, poi_id, pois):
    c_action = 0.0

    if pois[poi_id, 2] > 5.0:
        c_action = 100.00
    else:
        c_action = 1.0

    return c_action


def low_val_action_suggestions(rover_dist, poi_id, pois):
    c_action = 0.0

    if pois[poi_id, 2] > 5.0:
        c_action = 100.00
    else:
        c_action = 1.0

    return c_action


def high_low_actions(rover_dist, rover_id, poi_id, pois):
    c_action = 0.0

    if rover_id % 2 == 0:
        if pois[poi_id, 2] > 5.0:
            c_action = 100.00
        else:
            c_action = rover_dist
    else:
        if pois[poi_id, 2] <= 5.0:
            c_action = 100.00
        else:
            c_action = rover_dist

    return c_action


def three_rov_three_poi(rover_dist, rover_id, poi_id, pois):
    c_action = 0.0

    if rover_id == 0:
        if pois[poi_id, 2] > 5.0 and pois[poi_id, 2] < 10.0:
            c_action = 100.00
        else:
            c_action = 1.0
    elif rover_id == 1:
        if pois[poi_id, 2] <= 5.0:
            c_action = 100.00
        else:
            c_action = 1.0
    else:
        if pois[poi_id, 2] >= 10.0:
            c_action = 100.00
        else:
            c_action = 1.0

    return c_action


def three_rov_three_poi_internal(rover_id, poi_id, pois):
    c_action = 0.0

    if rover_id == 0:
        if pois[poi_id, 2] > 5.0 and pois[poi_id, 2] < 10.0:
            c_action = 100.00
        else:
            c_action = 1.0
    elif rover_id == 1:
        if pois[poi_id, 2] <= 5.0:
            c_action = 100.00
        else:
            c_action = 1.0
    else:
        if pois[poi_id, 2] >= 10.0:
            c_action = 100.00
        else:
            c_action = 1.0

    return c_action


def get_counterfactual_action(rover_dist, rov_id, poi_id, pois, suggestion):
    c_action = 0.0

    if suggestion == "high_val":
        c_action = high_val_action_suggestions(rover_dist, poi_id, pois)
    elif suggestion == "low_val":
        c_action = low_val_action_suggestions(rover_dist, poi_id, pois)
    elif suggestion == "3R3P":
        c_action = three_rov_three_poi(rover_dist, rov_id, poi_id, pois)
    elif suggestion == "high_low":
        high_low_actions(rover_dist, rov_id, poi_id, pois)

    return c_action
