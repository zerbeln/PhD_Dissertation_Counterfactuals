from parameters import parameters as p


def towards_teammates_reward(rovers, rover_id):
    """
    Rovers receive a reward for travelling towards other rovers
    """

    rov_x = rovers["Rover{0}".format(rover_id)].x_pos
    rov_y = rovers["Rover{0}".format(rover_id)].y_pos

    reward = 0
    for rov in rovers:
        if rov.self_id != rover_id:
            x = rov.x_pos
            y = rov.y_pos
            dist = (rov_x - x)**2 + (rov_y - y)**2

            reward -= dist

    return reward


def away_teammates_reward(rovers, rover_id):
    """
    Rovers receive a reward for travelling away from other rovers
    """
    rov_x = rovers["Rover{0}".format(rover_id)].x_pos
    rov_y = rovers["Rover{0}".format(rover_id)].y_pos

    reward = 0
    for rov in rovers:
        if rov.self_id != rover_id:
            x = rov.x_pos
            y = rov.y_pos
            dist = (rov_x - x) ** 2 + (rov_y - y) ** 2

            reward += dist

    return reward


def towards_poi_reward(rover_id, pois):
    """
    Rovers receive a reward for travelling towards POI
    """
    reward = 0
    for poi in pois:
        dist = poi.observer_distances[rover_id]

        reward -= dist

    return reward


def away_poi_reward(rover_id, pois):
    """
    Rovers receive a reward for travelling away from POI
    """
    reward = 0
    for poi in pois:
        dist = poi.observer_distances[rover_id]

        reward += dist

    return reward


def greedy_reward_loose(rover_id, pois):
    """
    Greedy local reward for rovers
    """
    obs_rad = p["observation_radius"]
    reward = 0

    for poi in pois:
        dist = poi.observer_distances[rover_id]

        if dist < obs_rad:
            reward += poi.value / dist

    return reward


def target_poi_reward(rover_id, pois, target_poi):
    """
    Local rewards for going towards either the left or right POI
    """

    reward = 0

    dist = pois["P{0}".format(target_poi)].observer_distances[rover_id]

    if dist < p["observation_radius"]:
        reward += pois["P{0}".format(target_poi)].value/dist

    return reward


def target_quadrant_reward(rover_id, pois, target_quadrant):
    """
    Local reward for training rovers to travel towards POI within a specific quadrant
    """
    reward = 0
    for poi in pois:
        if target_quadrant == pois[poi].quadrant:
            dist = pois[poi].observer_distances[rover_id]

            if dist < p["observation_radius"]:
                reward += pois[poi].value / dist

    return reward

