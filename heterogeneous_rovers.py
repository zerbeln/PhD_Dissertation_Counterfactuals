from parameters import Parameters as p
import numpy as np
import random

def init_rover_types():
    nrovers = (p.num_rovers * p.num_types)
    rover_positions = np.zeros((nrovers, 3))

    for rov_id in range(nrovers):
        rover_positions[rov_id, 0] = 0.5*p.world_size  # Rover X-Coordinate
        rover_positions[rov_id, 1] = 0.5*p.world_size  # Rover Y-Coordinate
        rover_positions[rov_id, 2] = random.randint(0, (p.num_types-1))  # Rover type

    return rover_positions

def init_poi_positions():
    poi_positions = np.zeros((p.num_pois, 2))

    for poi_id in range(p.num_pois):
        poi_positions[poi_id, 0] = random.uniform(0, p.world_size-1)
        poi_positions[poi_id, 1] = random.uniform(0, p.world_size-1)

    return poi_positions

def init_poi_values():  # POI values randomly assigned 1-10
    poi_vals = np.zeros(p.num_pois)

    for poi_id in range(p.num_pois):
        poi_vals[poi_id] = random.randint(1, 10)

    return poi_vals
