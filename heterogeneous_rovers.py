from parameters import Parameters as p
import numpy as np
import random

def init_rover_types():
    nrovers = p.num_rovers * p.num_types
    rover_positions = np.zeros((nrovers, 3))

    for rov_id in range(nrovers):
        rover_positions[rov_id, 0] = 0.5*p.world_size  # Rover X-Coordinate
        rover_positions[rov_id, 1] = 0.5*p.world_size  # Rover Y-Coordinate
        rover_positions[rov_id, 2] = random.randint(0, (p.num_types-1))  # Rover type

    return rover_positions

def init_poi_values():  # POI values randomly assigned 1-10
    poi_vals = np.zeros(p.num_pois)

    for poi_id in range(p.num_pois):
        poi_vals[poi_id] = random.randint(1, 10)

    return poi_vals
