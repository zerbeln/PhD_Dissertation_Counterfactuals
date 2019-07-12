import numpy as np

def homogeneous_partner_suggestions(npartners, rx, ry, poi_id, poi_positions, poi_values):
    partners = np.zeros((npartners, 2))

    if poi_values[poi_id] > 5:
        for partner_id in range(npartners):
            partners[partner_id, 0] = rx
            partners[partner_id, 1] = ry
    else:
        for partner_id in range(npartners):
            partners[partner_id, 0] = -100.0
            partners[partner_id, 1] = -100.0

    return partners
