from AADI_RoverDomain.parameters import Parameters as p
import numpy as np
import random

### ROVER SETUP FUNCTIONS ######################################################

def init_rover_positions_fixed():  # Set rovers to fixed starting position
    """
    Rovers all start out in the middle of the map at an orientation of 0 degrees.
    :return: rover_positions: np array of size (nrovers, 3)
    """
    nrovers = p.num_rovers
    rover_positions = np.zeros((nrovers, 3))

    for rov_id in range(p.num_rovers):
        rover_positions[rov_id, 0] = 0.5*p.x_dim  # Rover X-Coordinate
        rover_positions[rov_id, 1] = 0.5*p.y_dim  # Rover Y-Coordinate
        rover_positions[rov_id, 2] = 0.0  # Rover orientation

    return rover_positions

def init_rover_positions_random():  # Randomly set rovers on map
    """
    Rovers given random starting positions and orientations
    :return: rover_positions: np array of size (nrovers, 3)
    """
    nrovers = p.num_rovers
    rover_positions = np.zeros((nrovers, 3))

    for rov_id in range(p.num_rovers):
        rover_positions[rov_id, 0] = random.uniform(0, p.x_dim-1)  # Rover X-Coordinate
        rover_positions[rov_id, 1] = random.uniform(0, p.y_dim-1)  # Rover Y-Coordinate
        rover_positions[rov_id, 2] = random.uniform(0, 360)  # Rover orientation

    return rover_positions

### POI SETUP FUNCTIONS ###########################################################

def init_poi_positions_random():  # Randomly set POI on the map
    """
    POI positions set randomly across the map
    :return: poi_positions: np array of size (npoi, 2)
    """
    poi_positions = np.zeros((p.num_pois, 2))

    for poi_id in range(p.num_pois):
        poi_positions[poi_id, 0] = random.uniform(0, p.x_dim-1)
        poi_positions[poi_id, 1] = random.uniform(0, p.y_dim-1)

    return poi_positions

def init_poi_positions_two_poi():
    """
    Sets two POI on the map, one on the left, one on the right at Y-Dimension/2
    :return: poi_positions: np array of size (npoi, 2)
    """
    assert(p.num_pois == 2)

    poi_positions = np.zeros((p.num_pois, 2))

    poi_positions[0, 0] = 0.0; poi_positions[0, 1] = p.y_dim/2
    poi_positions[1, 0] = (p.x_dim-1); poi_positions[1, 1] = p.y_dim/2

    return poi_positions


def init_poi_positions_four_corners():  # Statically set 4 POI (one in each corner)
    """
    Sets 4 POI on the map, one in each corner
    :return: poi_positions: np array of size (npoi, 2)
    """
    assert(p.num_pois == 4)  # There must only be 4 POI for this initialization

    poi_positions = np.zeros((p.num_pois, 2))

    poi_positions[0, 0] = 0.0; poi_positions[0, 1] = 0.0  # Bottom left
    poi_positions[1, 0] = 0.0; poi_positions[1, 1] = (p.y_dim - 1.0)  # Top left
    poi_positions[2, 0] = (p.x_dim - 1.0); poi_positions[2, 1] = 0.0  # Bottom right
    poi_positions[3, 0] = (p.x_dim - 1.0); poi_positions[3, 1] = (p.y_dim - 1.0)  # Top right

    return poi_positions

def init_poi_positions_txt_file():
    """
    POI positions read in from existing txt file (TXT FILE NEEDED FOR THIS FUNCTION)
    :return: poi_positions: np array of size (npoi, 2)
    """
    poi_positions = np.zeros((p.num_pois, 2))

    with open('Output_Data/POI_Positions.txt') as f:
        for i, l in enumerate(f):
            pass

    line_count = i + 1

    posFile = open('Output_Data/POI_Positions.txt', 'r')

    count = 1
    coordMat = []

    for line in posFile:
        for coord in line.split('\t'):
            if (coord != '\n') and (count == line_count):
                coordMat.append(float(coord))
        count += 1

    prev_pos = np.reshape(coordMat, (p.num_pois, 2))

    for ii in range(p.num_pois):
        poi_positions[ii, 0] = prev_pos[ii, 0]
        poi_positions[ii, 1] = prev_pos[ii, 1]

    return poi_positions



def init_poi_values_random():
    """
    POI values randomly assigned 1-10
    :return: poi_vals: array of size(npoi)
    """
    poi_vals = [0.0 for _ in range(p.num_pois)]

    for poi_id in range(p.num_pois):
        poi_vals[poi_id] = random.randint(1, 10)

    return poi_vals


def init_poi_values_fixed():
    """
    POI values set to fixed value
    :return: poi_vals: array of size(npoi)
    """
    poi_vals = [1.0 for _ in range(p.num_pois)]

    for poi_id in range(p.num_pois):
        poi_vals[poi_id] = poi_vals[poi_id] * 5

    return poi_vals
