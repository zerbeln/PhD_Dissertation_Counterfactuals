import numpy as np
import sys

# COUNTERFACTUAL PARTNER SUGGESTIONS ---------------------------------------------------------------------------------
cpdef low_high_split(double rover_dist, int rover_id, int poi_id, double [:, :] pois, int n_counters, double obs_rad):
    """
    Rovers with even IDs go for high value POIs, Rovers with odd IDs go for low value POIs
    :param rover_dist:
    :param rover_id:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :return: partners
    """
    cdef int partner_id
    cdef double [:] partners = np.zeros(n_counters)

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

cpdef high_value_only(double rover_dist, int poi_id, double [:, :] pois, int n_counters):
    """
    Suggestions give rover stepping stone reward only for high value POIs
    :param rover_dist:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :return: partners
    """
    cdef int partner_id
    cdef double [:] partners = np.zeros(n_counters)

    if pois[poi_id, 2] > 5.0:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.00

    return partners

cpdef low_value_only(double rover_dist, int poi_id, double [:, :] pois, int n_counters):
    """
    Suggestions give rover stepping stone reward only for low value POIs
    :param rover_dist:
    :param poi_id:
    :param poi_values:
    :param n_counters:
    :return: partners
    """
    cdef int partner_id
    cdef double [:] partners = np.zeros(n_counters)

    if pois[poi_id, 2] <= 5.0:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.00

    return partners


cpdef value_based_incentives(double rover_dist, int poi_id, double [:, :] pois, int n_counters, double min_dist, double obs_rad):
    """
    :param rover_dist: 
    :param poi_id: 
    :param poi_values: 
    :param n_counters: 
    :param min_dist: 
    :param obs_rad: 
    :return: 
    """
    cdef int partner_id
    cdef double [:] partners = np.zeros(n_counters)

    if rover_dist < obs_rad:
        if pois[poi_id, 2] > 5:
            for partner_id in range(n_counters):
                partners[partner_id] = min_dist
        else:
            for partner_id in range(n_counters):
                partners[partner_id] = obs_rad - 0.01
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.0

    return partners

cpdef get_counterfactual_partners(int n_counters, int nrovers, int self_id, double rover_dist, double [:, :, :] rover_paths, int poi_id, double [:, :] pois, int step_id, str suggestion, double min_dist, double obs_rad):
    cdef int partner_id
    cdef double [:] partners = np.zeros(n_counters)

    if suggestion == "high_val":
        partners = high_value_only(rover_dist, poi_id, pois, n_counters)
    elif suggestion == "low_val":
        partners = low_value_only(rover_dist, poi_id, pois, n_counters)
    elif suggestion == "high_low":
        partners = low_high_split(rover_dist, self_id, poi_id, pois, n_counters, obs_rad)
    elif suggestion == "val_based":
        partners = value_based_incentives(rover_dist, poi_id, pois, n_counters, min_dist, obs_rad)
    else:
        sys.exit('Incorrect Suggestion Type')

    return partners


# COUNTERFACTUAL ACTION SUGGESTIONS -----------------------------------------------------------------------------------
cpdef high_val_action_suggestions(double rover_dist, int poi_id, double[:, :] pois):
    cdef double c_action = 0.0

    if pois[poi_id, 2] > 5.0 or rover_dist > 3.0:
        c_action = 100.00
    else:
        c_action = 1.0

    return c_action

cpdef low_val_action_suggestions(double rover_dist, int poi_id, double[:, :] pois):
    cdef double c_action = 0.0

    if pois[poi_id, 2] > 5.0 or rover_dist > 3.0:
        c_action = 100.00
    else:
        c_action = 1.0

    return c_action

cpdef get_counterfactual_action(double rover_dist, int poi_id, double[:, :] pois, str suggestion):
    cdef double c_action = 0.0

    if suggestion == "high_val":
        c_action = high_val_action_suggestions(rover_dist, poi_id, pois)
    elif suggestion == "low_val":
        c_action = low_val_action_suggestions(rover_dist, poi_id, pois)

    return c_action
