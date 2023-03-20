import math
from parameters import parameters as p
import numpy as np


def sigmoid(inp):  # Sigmoid function as activation function
    """
    sigmoid neural network activation function
    """
    sig = 1 / (1 + np.exp(-inp))

    return sig


def get_custom_action(chosen_pol, pois, rover_x, rover_y):
    """
    Rover receives a customized (hard-coded) action
    """
    if chosen_pol < p["n_poi"]:
        action = travel_to_poi(chosen_pol, pois, rover_x, rover_y)
    else:
        action = remain_stationary()

    return action


def travel_to_poi(target_poi, pois, rover_x, rover_y):
    """
    Prouces an action that will make a rover travel in a straight line towards a POI
    """
    poi_x = pois["P{0}".format(target_poi)].loc[0]
    poi_y = pois["P{0}".format(target_poi)].loc[1]

    delta_x = poi_x - rover_x
    delta_y = poi_y - rover_y

    vector_mag = math.sqrt(delta_x**2 + delta_y**2)

    if vector_mag >= 1:
        theta = math.atan2(delta_y, delta_x)
        dx = sigmoid(math.cos(theta))
        dy = sigmoid(math.sin(theta))
    else:
        dx = 0.5
        dy = 0.5

    return [dx, dy]


def travel_north():
    """
    Rover moves "up"
    """
    return [0.5, 1]


def travel_south():
    """
    Rover moves "down"
    """
    return [0.5, 0]


def travel_east():
    """
    Rover moves "right"
    """
    return [1, 0.5]


def travel_west():
    """
    Rover moves "left"
    """
    return [0, 0.5]


def remain_stationary():
    """
    Rover does not move
    """
    return [0.5, 0.5]
