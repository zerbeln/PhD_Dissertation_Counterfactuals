from AADI_RoverDomain.parameters import Parameters as p
import numpy as np
import os


def generate_poi_positions():
    poi_positions = np.zeros((p.num_pois, 2))

    dir_name = 'Output_Data/'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    pcoords_name = os.path.join(dir_name, 'POI_Positions.txt')

    for poi_id in range(p.num_pois):
        poi_positions[poi_id, 0] = np.random.uniform(0, p.x_dim-1)
        poi_positions[poi_id, 1] = np.random.uniform(0, p.y_dim-1)

    poi_coords = open(pcoords_name, 'w')
    for p_id in range(p.num_pois):  # Record POI positions and values
        poi_coords.write('%f' % poi_positions[p_id, 0])
        poi_coords.write('\t')
        poi_coords.write('%f' % poi_positions[p_id, 1])
        poi_coords.write('\t')
    poi_coords.write('\n')
    poi_coords.close()

generate_poi_positions()
