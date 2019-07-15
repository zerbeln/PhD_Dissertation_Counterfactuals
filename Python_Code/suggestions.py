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

def high_reward_four_corner_suggestions(npartners, rx, ry, rover_id, poi_id, poi_positions, poi_values):
    partners = np.zeros((npartners, 2))

    if poi_values[poi_id] == 100:
        for partner_id in range(npartners):
            partners[partner_id, 0] = poi_positions[poi_id, 0] + 0.5
            partners[partner_id, 1] = poi_positions[poi_id, 1]
    else:
        for partner_id in range(npartners):
            partners[partner_id, 0] = rx
            partners[partner_id, 1] = ry

    return partners

def partner_distance(npartners, rx, ry, rover_id, poi_id, poi_positions, poi_values):
    partners = np.zeros(npartners)

    if poi_values[poi_id] == 100:
        for partner_id in range(npartners):
            partners[partner_id] = p.min_distance
    else:
        for partner_id in range(npartners):
            partners[partner_id] = 100.00

    return partners