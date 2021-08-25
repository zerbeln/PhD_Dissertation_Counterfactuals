from parameters import parameters as p


def towards_teammates_reward(rovers, rover_id):
    """
    Rovers receive a reward for travelling towards other rovers
    """

    n_rovers = p["n_rovers"]
    rov_x = rovers["Rover{0}".format(rover_id)].pos[0]
    rov_y = rovers["Rover{0}".format(rover_id)].pos[1]

    reward = 0
    for agent_id in range(n_rovers):
        if agent_id != rover_id:
            x = rovers["Rover{0}".format(agent_id)].pos[0]
            y = rovers["Rover{0}".format(agent_id)].pos[1]
            dist = (rov_x - x)**2 + (rov_y - y)**2

            reward -= dist

    return reward


def away_teammates_reward(rovers, rover_id):
    """
    Rovers receive a reward for travelling away from other rovers
    """

    n_rovers = p["n_rovers"]
    rov_x = rovers["Rover{0}".format(rover_id)].pos[0]
    rov_y = rovers["Rover{0}".format(rover_id)].pos[1]

    reward = 0
    for agent_id in range(n_rovers):
        if agent_id != rover_id:
            x = rovers["Rover{0}".format(agent_id)].pos[0]
            y = rovers["Rover{0}".format(agent_id)].pos[1]
            dist = (rov_x - x) ** 2 + (rov_y - y) ** 2

            reward += dist

    return reward


def towards_poi_reward(rover_id, observer_distances):
    """
    Rovers receive a reward for travelling towards POI
    """

    n_poi = p["n_poi"]
    reward = 0
    for poi_id in range(n_poi):
        dist = observer_distances[poi_id, rover_id]

        reward -= dist

    return reward


def away_poi_reward(rover_id, observer_distances):
    """
    Rovers receive a reward for travelling away from POI
    """

    n_poi = p["n_poi"]
    reward = 0
    for poi_id in range(n_poi):
        dist = observer_distances[poi_id, rover_id]

        reward += dist

    return reward


def greedy_reward_loose(rover_id, observer_distances, poi):
    """
    Greedy local reward for rovers
    """
    n_poi = p["n_poi"]
    obs_rad = p["observation_radius"]
    reward = 0

    for poi_id in range(n_poi):
        dist = observer_distances[poi_id, rover_id]

        if dist < obs_rad:
            reward += poi[poi_id, 2] / dist

    return reward
