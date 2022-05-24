import math


def get_angle(target_x, target_y, x, y):
    """
    Calculates the angle between line connecting a target (target_x, target_y) and an origin point (x, y) with respect
    to the x-axis
    """
    dx = target_x - x
    dy = target_y - y

    angle = math.atan2(dy, dx) * (180.0 / math.pi)
    while angle < 0.0:
        angle += 360.0
    while angle > 360.0:
        angle -= 360.0
    if math.isnan(angle):
        angle = 0.0

    return angle


def get_squared_dist(target_x, target_y, x, y):
    """
    Calculates the squared distance between a target (target_x, target_y) and an origin point (x, y)
    """

    dx = target_x - x
    dy = target_y - y

    dist = (dx ** 2) + (dy ** 2)

    return dist


def get_linear_dist(target_x, target_y, x, y):
    """
    Calculates the linear distance between a target (target_x, target_y) and an origin point (x, y)
    """

    dx = target_x - x
    dy = target_y - y

    dist = math.sqrt((dx ** 2) + (dy ** 2))


    return dist
