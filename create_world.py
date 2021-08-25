from Python_Code.rover_domain import RoverDomain
from Python_Code.agent import Rover
from Visualizer.visualizer import run_visualizer
import numpy as np
from parameters import parameters as p


if __name__ == '__main__':
    """
    Create new world configuration files for POI and rovers
    """

    stat_runs = p["stat_runs"]
    rover_path = np.zeros((stat_runs, p["n_rovers"], p["steps"], 3))

    for srun in range(stat_runs):
        rd = RoverDomain()  # Number of POI, Number of Rovers
        rd.create_world_setup(srun)
        rovers = {}
        for rover_id in range(p["n_rovers"]):
            rovers["Rover{0}".format(rover_id)] = Rover(rover_id)
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)

        for rover_id in range(p["n_rovers"]):
            for step in range(p["steps"]):
                rover_path[srun, rover_id, step, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                rover_path[srun, rover_id, step, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                rover_path[srun, rover_id, step, 2] = rovers["Rover{0}".format(rover_id)].pos[2]

    # save_rover_path(rover_path, "Rover_Paths")
    # run_visualizer()