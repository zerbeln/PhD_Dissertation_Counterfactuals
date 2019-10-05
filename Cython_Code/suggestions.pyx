import numpy as np
import math
import sys

cpdef low_high_split(double rover_dist, int rover_id, int poi_id, double [:] poi_values, int n_counters):
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

cpdef high_value_only(double rover_dist, int poi_id, double [:] poi_values, int n_counters):
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

    if poi_values[poi_id] > 5.0:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.0

    return partners

cpdef low_value_only(double rover_dist, int poi_id, double [:] poi_values, int n_counters):
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

    if poi_values[poi_id] <= 5.0:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.0

    return partners


cpdef value_based_incentives(double rover_dist, int poi_id, double [:] poi_values, int n_counters, double min_dist, double obs_rad):
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
        if poi_values[poi_id] > 5:
            for partner_id in range(n_counters):
                partners[partner_id] = min_dist
        else:
            for partner_id in range(n_counters):
                partners[partner_id] = obs_rad - 0.01
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.0

    return partners

cpdef partner_proximity_suggestions(double rover_dist, int n_counters, int self_id, double [:, :, :] rover_paths, int nrovers, int step_id, double min_dist, double obs_rad):
    """
    :param rover_dist: 
    :param n_counters: 
    :param self_id: 
    :param rover_paths: 
    :param nrovers: 
    :param step_id: 
    :param min_dist: 
    :param obs_rad: 
    :return: 
    """
    cdef int partner_id, count
    cdef double x_dist, y_dist, rover_x, rover_y, dist, self_x, self_y
    cdef double [:] partners = np.zeros(n_counters)
    count = 0

    # Calculate distance between self and other rovers
    self_x = rover_paths[step_id, self_id, 0]
    self_y = rover_paths[step_id, self_id, 1]
    for rover_id in range(nrovers):
        if rover_id == self_id:
            continue
        rover_x = rover_paths[step_id, rover_id, 0]
        rover_y = rover_paths[step_id, rover_id, 1]
        x_dist = rover_x - self_x
        y_dist = rover_y - self_y
        dist = math.sqrt((x_dist**2)+(y_dist**2))
        if dist < obs_rad:
            count += 1

    if count > 0:
        if rover_dist < obs_rad:
            for partner_id in range(n_counters):
                partners[partner_id] = min_dist
        else:
            for partner_id in range(n_counters):
                partners[partner_id] = obs_rad - 0.01
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = min_dist

    return partners

cpdef go_left_suggestions(double rover_dist, int poi_id, double [:, :] poi_pos, int n_counters, double xd):
    cdef int partner_id
    cdef double [:] partners = np.zeros(n_counters)
    cdef double x_middle

    partners = np.zeros(n_counters)
    x_middle = xd/2

    if poi_pos[poi_id, 0] < x_middle:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.00

    return partners

cpdef go_right_suggestions(double rover_dist, int poi_id, double [:, :] poi_pos, int n_counters, double xd):
    cdef int partner_id
    cdef double [:] partners = np.zeros(n_counters)
    cdef double x_middle

    partners = np.zeros(n_counters)
    x_middle = xd/2

    if poi_pos[poi_id, 0] > x_middle:
        for partner_id in range(n_counters):
            partners[partner_id] = rover_dist
    else:
        for partner_id in range(n_counters):
            partners[partner_id] = 100.00

    return partners

cpdef left_right_split(double rover_dist, int rover_id, int poi_id, double [:, :] poi_pos, int n_counters, double xd):
    cdef int partner_id
    cdef double [:] partners = np.zeros(n_counters)
    cdef double x_middle

    partners = np.zeros(n_counters)
    x_middle = xd/2

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

cpdef get_counterfactual_partners(int n_counters, int nrovers, int self_id, double rover_dist, double [:, :, :] rover_paths, int poi_id, double [:] poi_values, int step_id, str suggestion, double min_dist, double obs_rad):
    cdef int partner_id
    cdef double [:] partners = np.zeros(n_counters)

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
        partners = value_based_incentives(rover_dist, poi_id, poi_values, n_counters, min_dist, obs_rad)
    elif suggestion == "partner_proximity":
        partners = partner_proximity_suggestions(rover_dist, n_counters, self_id, rover_paths, nrovers, step_id, min_dist, obs_rad)
    else:
        sys.exit('Incorrect Suggestion Type')

    return partners
