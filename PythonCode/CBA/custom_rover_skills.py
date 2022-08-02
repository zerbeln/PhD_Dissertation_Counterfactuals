import math


def get_custom_action(chosen_pol, pois, rover_x, rover_y):
    n_poi = len(pois)

    if chosen_pol < n_poi:
        action = travel_to_poi(chosen_pol, pois, rover_x, rover_y)
    else:
        action = remain_stationary()

    return action


def travel_to_poi(target_poi, pois, rover_x, rover_y):

    poi_x = pois["P{0}".format(target_poi)].loc[0]
    poi_y = pois["P{0}".format(target_poi)].loc[1]

    delta_x = poi_x - rover_x
    delta_y = poi_y - rover_y

    vector_mag = math.sqrt(delta_x**2 + delta_y**2)

    if vector_mag >= 1:
        theta = math.atan2(delta_y, delta_x)
        dx = math.cos(theta)
        dy = math.sin(theta)
    else:
        dx = 0.0
        dy = 0.0

    return [dx, dy]


def travel_north():

    return [0, 1]


def travel_south():

    return [0, -1]


def travel_east():

    return [1, 0]


def travel_west():

    return [-1, 0]


def remain_stationary():

    return [0, 0]
