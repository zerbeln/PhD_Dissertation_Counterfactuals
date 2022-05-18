from RoverDomain_Core.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
import numpy as np
from parameters import parameters as p
import os
import pickle


def save_rover_path(rover_path, file_name):  # Save path rovers take using best policy found
    """
    Records the path each rover takes using best policy from CCEA (used by visualizer)
    :param rover_path:  trajectory tracker
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    rpath_name = os.path.join(dir_name, file_name)
    rover_file = open(rpath_name, 'wb')
    pickle.dump(rover_path, rover_file)
    rover_file.close()


if __name__ == '__main__':
    """
    Create new world configuration files for POI and rovers
    """

    coupling = 1  # Coupling requirement for POI
    stat_runs = p["stat_runs"]
    rover_path = np.zeros((stat_runs, p["n_rovers"], p["steps"], 3))

    rd = RoverDomain()  # Number of POI, Number of Rovers
    rd.create_world_setup(coupling)
    rd.load_world()

    for rover_id in range(p["n_rovers"]):
        for step in range(p["steps"]):
            rover_path[0:stat_runs, rover_id, step, 0] = rd.rovers["R{0}".format(rover_id)].x_pos
            rover_path[0:stat_runs, rover_id, step, 1] = rd.rovers["R{0}".format(rover_id)].y_pos
            rover_path[0:stat_runs, rover_id, step, 2] = rd.rovers["R{0}".format(rover_id)].theta_pos

    save_rover_path(rover_path, "Rover_Paths")
    run_visualizer()
