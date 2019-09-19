from AADI_RoverDomain.parameters import Parameters as p
import numpy as np
import random
import math

### ROVER SETUP FUNCTIONS ######################################################

def init_rover_positions_fixed_middle():  # Set rovers to fixed starting position
    """
    Rovers all start out in the middle of the map at random orientations
    :return: rover_positions: np array of size (nrovers, 3)
    """
    nrovers = p.num_rovers
    rover_positions = np.zeros((nrovers, 3))

    for rov_id in range(p.num_rovers):
        rover_positions[rov_id, 0] = 0.5*p.x_dim  # Rover X-Coordinate
        rover_positions[rov_id, 1] = 0.5*p.y_dim  # Rover Y-Coordinate
        rover_positions[rov_id, 2] = random.uniform(0, 360)  # Rover orientation

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

def init_rover_positions_random_concentrated():
    """
        Rovers given random starting positions within a radius of the center. Starting orientations are random
        :return: rover_positions: np array of size (nrovers, 3)
    """
    nrovers = p.num_rovers
    rover_positions = np.zeros((nrovers, 3))
    radius = 4.0; center_x = p.x_dim/2; center_y = p.y_dim/2

    for rov_id in range(p.num_rovers):
        x = random.uniform(0, p.x_dim)  # Rover X-Coordinate
        y = random.uniform(0, p.y_dim)  # Rover Y-Coordinate

        while x > (center_x + radius) or x < (center_x - radius):
            x = random.uniform(0, p.x_dim)  # Rover X-Coordinate
        while y > (center_y + radius) or y < (center_y - radius):
            y = random.uniform(0, p.y_dim)  # Rover Y-Coordinate

        rover_positions[rov_id, 0] = x  # Rover X-Coordinate
        rover_positions[rov_id, 1] = y  # Rover Y-Coordinate
        rover_positions[rov_id, 2] = random.uniform(0, 360)  # Rover orientation

    return rover_positions

def init_rover_pos_twelve_grid():
    rover_positions = np.zeros((p.num_rovers, 3))

    for rover_id in range(p.num_rovers):
        rover_positions[rover_id, 0] = p.x_dim - 1
        rover_positions[rover_id, 1] = random.uniform((p.y_dim / 2) - 5, (p.y_dim / 2) + 5)
        rover_positions[rover_id, 2] = random.uniform(0, 360)  # Rover orientation

    return rover_positions

def init_rover_pos_txt_file():
    rover_positions = np.zeros((p.num_rovers, 3))

    with open('Output_Data/Rover_Positions.txt') as f:
        for i, l in enumerate(f):
            pass

    line_count = i + 1

    posFile = open('Output_Data/Rover_Positions.txt', 'r')

    count = 1
    coordMat = []

    for line in posFile:
        for coord in line.split('\t'):
            if (coord != '\n') and (count == line_count):
                coordMat.append(float(coord))
        count += 1

    prev_pos = np.reshape(coordMat, (p.num_rovers, 3))

    for rover_id in range(p.num_rovers):
        rover_positions[rover_id, 0] = prev_pos[rover_id, 0]
        rover_positions[rover_id, 1] = prev_pos[rover_id, 1]
        rover_positions[rover_id, 2] = prev_pos[rover_id, 2]

    return rover_positions


### POI SETUP FUNCTIONS ###########################################################

def init_poi_positions_random(rover_positions):  # Randomly set POI on the map
    """
    POI positions set randomly across the map (but not in range of any rover)
    :return: poi_positions: np array of size (npoi, 2)
    """
    poi_positions = np.zeros((p.num_pois, 2))

    for poi_id in range(p.num_pois):
        x = random.uniform(0, p.x_dim-1)
        y = random.uniform(0, p.y_dim-1)

        rover_id = 0
        while rover_id < p.num_rovers:
            rovx = rover_positions[rover_id, 0]; rovy = rover_positions[rover_id, 1]
            xdist = x - rovx; ydist = y - rovy
            distance = math.sqrt((xdist**2) + (ydist**2))

            while distance < p.min_observation_dist:
                x = random.uniform(0, p.x_dim - 1)
                y = random.uniform(0, p.y_dim - 1)
                rovx = rover_positions[rover_id, 0]; rovy = rover_positions[rover_id, 1]
                xdist = x - rovx; ydist = y - rovy
                distance = math.sqrt((xdist ** 2) + (ydist ** 2))
                rover_id = -1

            rover_id += 1

        poi_positions[poi_id, 0] = x
        poi_positions[poi_id, 1] = y

    return poi_positions

def init_poi_positions_circle():
    """
        POI positions are set in a circle around the center of the map at a specified radius.
        :return: poi_positions: np array of size (npoi, 2)
    """
    radius = 15.0
    interval = 360/p.num_pois

    poi_positions = np.zeros((p.num_pois, 2))

    x = p.x_dim/2
    y = p.y_dim/2
    theta = 0.0

    for poi_id in range(p.num_pois):
        poi_positions[poi_id, 0] = x + radius*math.cos(theta*math.pi/180)
        poi_positions[poi_id, 1] = y + radius*math.sin(theta*math.pi/180)
        theta += interval


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
    Sets 4 POI on the map in a box formation around the center
    :return: poi_positions: np array of size (npoi, 2)
    """
    assert(p.num_pois == 4)  # There must only be 4 POI for this initialization

    poi_positions = np.zeros((p.num_pois, 2))

    poi_positions[0, 0] = 2.0; poi_positions[0, 1] = 2.0  # Bottom left
    poi_positions[1, 0] = 2.0; poi_positions[1, 1] = (p.y_dim - 2.0)  # Top left
    poi_positions[2, 0] = (p.x_dim - 2.0); poi_positions[2, 1] = 2.0  # Bottom right
    poi_positions[3, 0] = (p.x_dim - 2.0); poi_positions[3, 1] = (p.y_dim - 2.0)  # Top right

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

    for poi_id in range(p.num_pois):
        poi_positions[poi_id, 0] = prev_pos[poi_id, 0]
        poi_positions[poi_id, 1] = prev_pos[poi_id, 1]

    return poi_positions

def init_poi_values_txt_file():

    poi_vals = np.zeros(p.num_pois)

    poi_val_file = open('Output_Data/POI_Values.txt', 'r')

    value_mat = []
    for line in poi_val_file:
        for v in line.split('\t'):
            if v != '\n':
                value_mat.append(float(v))

    values = np.reshape(value_mat, p.num_pois)
    for poi_id in range(p.num_pois):
        poi_vals[poi_id] = values[poi_id]

    return poi_vals


def init_poi_values_random():
    """
    POI values randomly assigned 1-10
    :return: poi_vals: array of size(npoi)
    """
    poi_vals = np.zeros(p.num_pois)

    for poi_id in range(p.num_pois):
        poi_vals[poi_id] = random.randint(1, 10)

    return poi_vals


def init_poi_values_fixed_ascending():
    """
    POI values set to fixed, ascending values based on POI ID
    :return: poi_vals: array of size(npoi)
    """
    poi_vals = np.zeros(p.num_pois)

    for poi_id in range(p.num_pois):
        poi_vals[poi_id] = poi_id + 1

    return poi_vals

def init_poi_values_fixed_identical():
    """
        POI values set to fixed, identical value
        :return: poi_vals: array of size(npoi)
        """
    poi_vals = np.zeros(p.num_pois)

    for poi_id in range(p.num_pois):
        poi_vals[poi_id] = 5.0

    return poi_vals

def init_poi_values_half_and_half():
    """
    POI values set to fixed value
    :return: poi_vals: array of size(npoi)
    """

    poi_vals = np.ones(p.num_pois)

    for poi_id in range(p.num_pois):
        if poi_id%2 == 0:
            poi_vals[poi_id] *= 10.0
        else:
            poi_vals[poi_id] *= 5.0

    return poi_vals

def init_poi_pos_random_inner_square_outer():
    num_pois = p.num_pois
    num_outer_pois = 4
    num_inner_pois = num_pois - num_outer_pois

    poi_positions = np.zeros((p.num_pois, 2))

    poi_positions[0, 0] = 0.0; poi_positions[0, 1] = 0.0  # Bottom left
    poi_positions[1, 0] = 0.0; poi_positions[1, 1] = (p.y_dim - 1.0)  # Top left
    poi_positions[2, 0] = (p.x_dim - 1.0); poi_positions[2, 1] = 0.0  # Bottom right
    poi_positions[3, 0] = (p.x_dim - 1.0); poi_positions[3, 1] = (p.y_dim - 1.0)  # Top right

    for i in range(4, num_pois):
        poi_positions[i, 0] = random.uniform(p.x_dim / 4, 3 * p.x_dim / 4)
        poi_positions[i, 1] = random.uniform(p.y_dim / 4, 3 * p.y_dim / 4)

    return poi_positions

def init_poi_val_random_inner_square_outer():
    poi_vals = np.zeros(p.num_pois)

    for poi_id in range(4):
        poi_vals[poi_id] = 100

    for poi_id in range(4, p.num_pois):
        poi_vals[poi_id] = 5.0

    return poi_vals


def init_poi_pos_twelve_grid():

    assert(p.num_pois == 12)

    poi_positions = np.zeros((p.num_pois, 2))
    poi_id = 0

    for i in range(4):
        for j in range(3):
            poi_positions[poi_id, 0] = j * ((p.x_dim - 10) / 2)
            poi_positions[poi_id, 1] = i * (p.y_dim / 3)

            poi_id += 1

    return poi_positions


def init_poi_concentric_squares_pos():

    assert(p.num_pois == 8)

    poi_positions = np.zeros((p.num_pois, 2))

    poi_positions[0, 0] = (p.x_dim / 2) - 5
    poi_positions[0, 1] = (p.y_dim / 2) - 5

    poi_positions[1, 0] = (p.x_dim / 2) + 5
    poi_positions[1, 1] = (p.y_dim / 2) - 5

    poi_positions[2, 0] = (p.x_dim / 2) - 5
    poi_positions[2, 1] = (p.y_dim / 2) + 5

    poi_positions[3, 0] = (p.x_dim / 2) + 5
    poi_positions[3, 1] = (p.y_dim / 2) + 5

    poi_positions[4, 0] = (p.x_dim / 2) - 10
    poi_positions[4, 1] = (p.y_dim / 2) - 10

    poi_positions[5, 0] = (p.x_dim / 2) + 10
    poi_positions[5, 1] = (p.y_dim / 2) - 10

    poi_positions[6, 0] = (p.x_dim / 2) - 10
    poi_positions[6, 1] = (p.y_dim / 2) + 10

    poi_positions[7, 0] = (p.x_dim / 2) + 10
    poi_positions[7, 1] = (p.y_dim / 2) + 10

    return poi_positions

def init_poi_concentric_squares_vals():

    assert(p.num_pois == 8)
    poi_vals = np.zeros(p.num_pois)

    poi_vals[0] = 4
    poi_vals[1] = 4
    poi_vals[2] = 4
    poi_vals[3] = 4

    poi_vals[4] = 20
    poi_vals[5] = 20
    poi_vals[6] = 20
    poi_vals[7] = 20

    return poi_vals
