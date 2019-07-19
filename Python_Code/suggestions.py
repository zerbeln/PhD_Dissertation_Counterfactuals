import numpy as np
from AADI_RoverDomain.parameters import Parameters as p

def homogeneous_partner_suggestions(npartners, rx, ry, rover_id, poi_id, poi_positions, poi_values):
    partners = np.zeros((npartners, 2))

    if rover_id%2 == 0:
        if poi_values[poi_id] > 5:
            for partner_id in range(npartners):
                partners[partner_id, 0] = rx
                partners[partner_id, 1] = ry
        else:
            for partner_id in range(npartners):
                partners[partner_id, 0] = 100.0
                partners[partner_id, 1] = 100.0
    else:
        if poi_values[poi_id] <= 5:
            for partner_id in range(npartners):
                partners[partner_id, 0] = rx
                partners[partner_id, 1] = ry
        else:
            for partner_id in range(npartners):
                partners[partner_id, 0] = 100.0
                partners[partner_id, 1] = 100.0

    return partners

def partner_distance(npartners, rover_dist, rover_id, poi_id, poi_values):
    partners = np.zeros(npartners)

    n_added = 0

    if rover_id%2 == 0:
        if poi_values[poi_id] > 5.0:
            for partner_id in range(npartners):
                partners[partner_id] = rover_dist
                if rover_dist < p.min_observation_dist:
                    n_added += 1
        else:
            for partner_id in range(npartners):
                partners[partner_id] = 100.0
                n_added -= 1
    else:
        if poi_values[poi_id] <= 5.0:
            for partner_id in range(npartners):
                partners[partner_id] = rover_dist
                if rover_dist < p.min_observation_dist:
                    n_added += 1
        else:
            for partner_id in range(npartners):
                partners[partner_id] = 100.0
                n_added -= 1

    return partners, n_added
