import numpy as np
import random
import math

### ROVER SETUP FUNCTIONS ######################################################

def init_rover_pos_fixed_middle(nrovers, xd, yd):  # Set rovers to fixed starting position
    """
    Rovers all start out in the middle of the map at random orientations
    :return: rover_positions: np array of size (nrovers, 3)
    """
    rover_positions = np.zeros((nrovers, 3))

    for rov_id in range(nrovers):
        rover_positions[rov_id, 0] = 0.5*xd  # Rover X-Coordinate
        rover_positions[rov_id, 1] = 0.5*yd  # Rover Y-Coordinate
        rover_positions[rov_id, 2] = random.uniform(0, 360)  # Rover orientation

    return rover_positions

def init_rover_pos_random(nrovers, xd, yd):  # Randomly set rovers on map
    """
    Rovers given random starting positions and orientations
    :return: rover_positions: np array of size (nrovers, 3)
    """
    rover_positions = np.zeros((nrovers, 3))

    for rov_id in range(nrovers):
        rover_positions[rov_id, 0] = random.uniform(0, xd-1)  # Rover X-Coordinate
        rover_positions[rov_id, 1] = random.uniform(0, yd-1)  # Rover Y-Coordinate
        rover_positions[rov_id, 2] = random.uniform(0, 360)  # Rover orientation

    return rover_positions

def init_rover_pos_random_concentrated(nrovers, xd, yd):
    """
    Rovers given random starting positions within a radius of the center. Starting orientations are random
    :return: rover_positions: np array of size (nrovers, 3)
    """
    rover_positions = np.zeros((nrovers, 3))
    radius = 4.0; center_x = xd/2; center_y = yd/2

    for rov_id in range(nrovers):
        x = random.uniform(0, xd)  # Rover X-Coordinate
        y = random.uniform(0, yd)  # Rover Y-Coordinate

        while x > (center_x + radius) or x < (center_x - radius):
            x = random.uniform(0, xd)  # Rover X-Coordinate
        while y > (center_y + radius) or y < (center_y - radius):
            y = random.uniform(0, yd)  # Rover Y-Coordinate

        rover_positions[rov_id, 0] = x  # Rover X-Coordinate
        rover_positions[rov_id, 1] = y  # Rover Y-Coordinate
        rover_positions[rov_id, 2] = random.uniform(0, 360)  # Rover orientation

    return rover_positions

def init_rover_pos_twelve_grid(nrovers, xd, yd):
    rover_positions = np.zeros((nrovers, 3))

    for rover_id in range(nrovers):
        rover_positions[rover_id, 0] = xd - 1
        rover_positions[rover_id, 1] = random.uniform((yd / 2) - 5, (yd / 2) + 5)
        rover_positions[rover_id, 2] = random.uniform(0, 360)  # Rover orientation

    return rover_positions

def init_rover_pos_txt_file(nrovers):
    rover_positions = np.zeros((nrovers, 3))

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

    prev_pos = np.reshape(coordMat, (nrovers, 3))

    for rover_id in range(nrovers):
        rover_positions[rover_id, 0] = prev_pos[rover_id, 0]
        rover_positions[rover_id, 1] = prev_pos[rover_id, 1]
        rover_positions[rover_id, 2] = prev_pos[rover_id, 2]

    return rover_positions


### POI SETUP FUNCTIONS ###########################################################

# POI POSITION FUNCTIONS ------------------------------------------------------------------------------------------
def init_poi_positions_random(num_pois, num_rovers, rover_positions, obs_rad, xd, yd):  # Randomly set POI on the map
    """
    POI positions set randomly across the map (but not in range of any rover)
    :return: poi_positions: np array of size (npoi, 2)
    """
    poi_positions = np.zeros((num_pois, 2))

    for poi_id in range(num_pois):
        x = random.uniform(0, xd-1)
        y = random.uniform(0, yd-1)

        rover_id = 0
        while rover_id < num_rovers:
            rovx = rover_positions[rover_id, 0]; rovy = rover_positions[rover_id, 1]
            xdist = x - rovx; ydist = y - rovy
            distance = math.sqrt((xdist**2) + (ydist**2))

            while distance < obs_rad:
                x = random.uniform(0, xd - 1)
                y = random.uniform(0, yd - 1)
                rovx = rover_positions[rover_id, 0]; rovy = rover_positions[rover_id, 1]
                xdist = x - rovx; ydist = y - rovy
                distance = math.sqrt((xdist ** 2) + (ydist ** 2))
                rover_id = -1

            rover_id += 1

        poi_positions[poi_id, 0] = x
        poi_positions[poi_id, 1] = y

    return poi_positions

def init_poi_pos_circle(num_pois, xd, yd):
    """
        POI positions are set in a circle around the center of the map at a specified radius.
        :return: poi_positions: np array of size (npoi, 2)
    """
    radius = 13.0
    interval = 360/num_pois

    poi_positions = np.zeros((num_pois, 2))

    x = xd/2
    y = yd/2
    theta = 0.0

    for poi_id in range(num_pois):
        poi_positions[poi_id, 0] = x + radius*math.cos(theta*math.pi/180)
        poi_positions[poi_id, 1] = y + radius*math.sin(theta*math.pi/180)
        theta += interval

    return poi_positions

def init_poi_pos_concentric_circles(num_pois, xd, yd):
    """
        POI positions are set in a circle around the center of the map at a specified radius.
        :return: poi_positions: np array of size (npoi, 2)
    """
    assert(num_pois == 12)
    inner_radius = 6.0
    outter_radius = 15.0
    interval = 360 / (num_pois/2)

    poi_positions = np.zeros((num_pois, 2))

    x = xd / 2
    y = yd / 2
    theta = 0.0

    for poi_id in range(num_pois):
        if poi_id == 6:
            theta = 0
        if poi_id < 6:
            poi_positions[poi_id, 0] = x + inner_radius * math.cos(theta * math.pi / 180)
            poi_positions[poi_id, 1] = y + inner_radius * math.sin(theta * math.pi / 180)
            theta += interval
        else:
            poi_positions[poi_id, 0] = x + outter_radius * math.cos(theta * math.pi / 180)
            poi_positions[poi_id, 1] = y + outter_radius * math.sin(theta * math.pi / 180)
            theta += interval

    return poi_positions


def init_poi_pos_two_poi(num_pois, xd, yd):
    """
    Sets two POI on the map, one on the left, one on the right at Y-Dimension/2
    :return: poi_positions: np array of size (npoi, 2)
    """
    assert(num_pois == 2)

    poi_positions = np.zeros((num_pois, 2))

    poi_positions[0, 0] = 0.0; poi_positions[0, 1] = yd/2
    poi_positions[1, 0] = (xd-1); poi_positions[1, 1] = yd/2

    return poi_positions


def init_poi_pos_four_corners(num_pois, xd, yd):  # Statically set 4 POI (one in each corner)
    """
    Sets 4 POI on the map in a box formation around the center
    :return: poi_positions: np array of size (npoi, 2)
    """
    assert(num_pois == 4)  # There must only be 4 POI for this initialization

    poi_positions = np.zeros((num_pois, 2))

    poi_positions[0, 0] = 2.0; poi_positions[0, 1] = 2.0  # Bottom left
    poi_positions[1, 0] = 2.0; poi_positions[1, 1] = (yd - 2.0)  # Top left
    poi_positions[2, 0] = (xd - 2.0); poi_positions[2, 1] = 2.0  # Bottom right
    poi_positions[3, 0] = (xd - 2.0); poi_positions[3, 1] = (yd - 2.0)  # Top right

    return poi_positions

def init_poi_positions_txt_file(num_pois):
    """
    POI positions read in from existing txt file (TXT FILE NEEDED FOR THIS FUNCTION)
    :return: poi_positions: np array of size (npoi, 2)
    """
    poi_positions = np.zeros((num_pois, 2))

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

    prev_pos = np.reshape(coordMat, (num_pois, 2))

    for poi_id in range(num_pois):
        poi_positions[poi_id, 0] = prev_pos[poi_id, 0]
        poi_positions[poi_id, 1] = prev_pos[poi_id, 1]

    return poi_positions

def init_poi_values_txt_file(num_pois):

    poi_vals = np.zeros(num_pois)

    poi_val_file = open('Output_Data/POI_Values.txt', 'r')

    value_mat = []
    for line in poi_val_file:
        for v in line.split('\t'):
            if v != '\n':
                value_mat.append(float(v))

    values = np.reshape(value_mat, num_pois)
    for poi_id in range(num_pois):
        poi_vals[poi_id] = values[poi_id]

    return poi_vals


def init_poi_pos_random_inner_square_outer(num_pois, xd, yd):
    num_pois = num_pois
    num_outer_pois = 4
    num_inner_pois = num_pois - num_outer_pois

    poi_positions = np.zeros((num_pois, 2))

    poi_positions[0, 0] = 0.0; poi_positions[0, 1] = 0.0  # Bottom left
    poi_positions[1, 0] = 0.0; poi_positions[1, 1] = (yd - 1.0)  # Top left
    poi_positions[2, 0] = (xd - 1.0); poi_positions[2, 1] = 0.0  # Bottom right
    poi_positions[3, 0] = (xd - 1.0); poi_positions[3, 1] = (yd - 1.0)  # Top right

    for i in range(4, num_pois):
        poi_positions[i, 0] = random.uniform(xd / 4, 3 * xd / 4)
        poi_positions[i, 1] = random.uniform(yd / 4, 3 * yd / 4)

    return poi_positions


def init_poi_pos_twelve_grid(num_pois, xd, yd):

    assert(num_pois == 12)

    poi_positions = np.zeros((num_pois, 2))
    poi_id = 0

    for i in range(4):
        for j in range(3):
            poi_positions[poi_id, 0] = j * ((xd - 10) / 2)
            poi_positions[poi_id, 1] = i * (yd / 3)

            poi_id += 1

    return poi_positions


def init_poi_pos_concentric_squares(num_pois, xd, yd):

    assert(num_pois == 8)

    poi_positions = np.zeros((num_pois, 2))

    # Inner-Bottom POI
    poi_positions[0, 0] = (xd / 2)
    poi_positions[0, 1] = (yd / 2) - 6

    # Inner-Right POI
    poi_positions[1, 0] = (xd / 2) + 6
    poi_positions[1, 1] = (yd / 2)

    # Inner-Top POI
    poi_positions[2, 0] = (xd / 2)
    poi_positions[2, 1] = (yd / 2) + 6

    # Inner-Left POI
    poi_positions[3, 0] = (xd / 2) - 6
    poi_positions[3, 1] = (yd / 2)

    # Outter-Bottom-Left POI
    poi_positions[4, 0] = (xd / 2) - 15
    poi_positions[4, 1] = (yd / 2) - 15

    # Outter-Bottom-Right POI
    poi_positions[5, 0] = (xd / 2) + 15
    poi_positions[5, 1] = (yd / 2) - 15

    # Outter-Top-Left POI
    poi_positions[6, 0] = (xd / 2) - 15
    poi_positions[6, 1] = (yd / 2) + 15

    # Outter-Top-Right POI
    poi_positions[7, 0] = (xd / 2) + 15
    poi_positions[7, 1] = (yd / 2) + 15

    return poi_positions

# POI VALUE FUNCTIONS -----------------------------------------------------------------------------------
def init_poi_vals_random(num_pois):
    """
    POI values randomly assigned 1-10
    :return: poi_vals: array of size(npoi)
    """
    poi_vals = np.zeros(num_pois)

    for poi_id in range(num_pois):
        poi_vals[poi_id] = random.randint(1, 10)

    return poi_vals


def init_poi_vals_fixed_ascending(num_pois):
    """
    POI values set to fixed, ascending values based on POI ID
    :return: poi_vals: array of size(npoi)
    """
    poi_vals = np.zeros(num_pois)

    for poi_id in range(num_pois):
        poi_vals[poi_id] = poi_id + 1

    return poi_vals

def init_poi_vals_fixed_identical(num_pois):
    """
        POI values set to fixed, identical value
        :return: poi_vals: array of size(npoi)
        """
    poi_vals = np.zeros(num_pois)

    for poi_id in range(num_pois):
        poi_vals[poi_id] = 10.0

    return poi_vals

def init_poi_vals_half_and_half(num_pois):
    """
    POI values set to fixed value
    :return: poi_vals: array of size(npoi)
    """

    poi_vals = np.ones(num_pois)

    for poi_id in range(num_pois):
        if poi_id%2 == 0:
            poi_vals[poi_id] *= 12.0
        else:
            poi_vals[poi_id] *= 5.0

    return poi_vals

def init_poi_vals_concentric_squares(num_pois):

    assert(num_pois == 8)
    poi_vals = np.zeros(num_pois)

    # Inner POI Values
    poi_vals[0] = 2.0
    poi_vals[1] = 2.0
    poi_vals[2] = 2.0
    poi_vals[3] = 2.0

    # Outter POI Values
    poi_vals[4] = 10.0
    poi_vals[5] = 10.0
    poi_vals[6] = 10.0
    poi_vals[7] = 10.0

    return poi_vals

def init_poi_vals_concentric_circles(num_pois):
    assert(num_pois == 12)

    poi_vals = np.zeros(num_pois)
    for poi_id in range(num_pois):
        if poi_id < 6:
            poi_vals[poi_id] = 2.0
        else:
            poi_vals[poi_id] = 12.0

    return poi_vals

def init_poi_vals_random_inner_square_outer(num_pois):
    poi_vals = np.zeros(num_pois)

    for poi_id in range(4):
        poi_vals[poi_id] = 100

    for poi_id in range(4, num_pois):
        poi_vals[poi_id] = 5.0

    return poi_vals

def init_poi_vals_four_corners(num_pois):
    poi_vals = np.zeros(num_pois)
    assert(num_pois == 4)

    for poi_id in range(num_pois):
        if poi_id == 0:
            poi_vals[poi_id] = 2.0
        elif poi_id == 1:
            poi_vals[poi_id] = 5.0
        elif poi_id == 2:
            poi_vals[poi_id] = 6.0
        else:
            poi_vals[poi_id] = 12.0

    return poi_vals
