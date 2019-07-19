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

def partner_distance(nobservers, rover_dist, rover_id, poi_id, poi_values):

    npartners = p.coupling - nobservers
    if npartners > 0:
        partners = np.zeros(npartners)

        if rover_id%2 == 0:
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
    else:
        partners = 0
        npartners = 0

    return partners, npartners
