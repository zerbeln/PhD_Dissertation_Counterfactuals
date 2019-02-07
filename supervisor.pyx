import math
cimport cython
from parameters import Parameters as p
import numpy as np
import random

cpdef closest_others(tstep, c_number, current_rov, current_poi, rover_hist, poi_pos):
    cdef double act_distance = p.activation_dist
    cdef double[:] rov_distances = np.zeros(p.num_rovers)
    cdef int[:] rov_ids = np.zeros(p.num_rovers, dtype = np.int32)
    cdef double[:] partners = np.zeros(c_number)
    cdef double[:, :, :] rover_positions = rover_hist
    cdef double[:, :] poi_positions = poi_pos
    cdef int agent_id, other_id, i, j
    cdef double agent_x, agent_y, other_agent_x, other_agent_y, x_dist, y_dist, dist

    agent_x = rover_positions[tstep, current_rov, 0]
    agent_y = rover_positions[tstep, current_rov, 1]
    for other_id in range(p.num_rovers):  # For current time step, figure out distance between current rover and others
        rov_ids[other_id] = other_id
        if current_rov != other_id:
            other_agent_x = rover_positions[tstep, other_id, 0]
            other_agent_y = rover_positions[tstep, other_id, 1]

            x_dist = agent_x - other_agent_x
            y_dist = agent_y - other_agent_y
            dist = math.sqrt((x_dist**2)+(y_dist**2))
            rov_distances[other_id] = dist
        if current_rov == other_id:  # Discounts self
            rov_distances[other_id] = 1000.00

    for i in range(len(rov_distances)):  # Finds closest other rovers to current rover
        j = i + 1
        while j < len(rov_distances):
            if rov_distances[i] > rov_distances[j]:
                rov_distances[i], rov_distances[j] = rov_distances[j], rov_distances[i]
                rov_ids[i], rov_ids[j] = rov_ids[j], rov_ids[i]
            j += 1

    for i in range(c_number):  # Computes distances from closest others to POI
        agent_id = rov_ids[i]
        dist_x = poi_positions[current_poi, 0] - rover_positions[tstep, other_id, 0]
        dist_y = poi_positions[current_poi, 1] - rover_positions[tstep, other_id, 1]
        dist = math.sqrt((dist_x**2) + (dist_y**2))
        if dist > p.activation_dist:  # If rover is out of range, suppose it is barely within range
            dist = p.activation_dist
        partners[i] = dist

    return partners

cpdef random_partners(tstep, c_number, current_rov, current_poi, rover_hist, poi_pos):
    cdef double n_rovers = p.num_rovers
    cdef double[:] partners = np.zeros(c_number)
    cdef double[:, :, :] rover_positions = rover_hist
    cdef double[:, :] poi_positions = poi_pos
    cdef int partner_id, i
    cdef double agent_x, agent_y, x_dist, y_dist, dist


    for i in range(c_number):  # Computes distances from closest others to POI
        partner_id = random.randint(0, n_rovers-1)
        while partner_id == current_rov:
            partner_id = random.randint(0, n_rovers-1)

        agent_x = rover_positions[tstep, partner_id, 0]
        agent_y = rover_positions[tstep, partner_id, 1]
        x_dist = poi_positions[current_poi, 0] - agent_x
        y_dist = poi_positions[current_poi, 1] - agent_y
        dist = math.sqrt((x_dist**2)+(y_dist**2))
        if dist < p.activation_dist:  # If rover is not in range, suppose that it is
            dist = p.activation_dist
        partners[i] = dist

    return partners


