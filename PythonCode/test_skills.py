from RoverDomain_Core.rover_domain import RoverDomain
from RewardFunctions.cba_rewards import *
from Visualizer.visualizer import run_visualizer
import numpy as np
from parameters import parameters as p
from custom_rover_skills import travel_to_poi
from global_functions import create_csv_file, create_pickle_file, load_saved_policies


def test_trained_skills(skill_id):
    """
    Test suggestions using the pre-trained policy bank
    """
    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    srun = p["starting_srun"]
    while srun < stat_runs:
        skill_performance = []  # Keep track of team performance throughout training

        # Load Trained Skills
        for rov in rd.rovers:
            rover_id = rd.rovers[rov].self_id
            if p["skill_type"] == "Target_POI":
                weights = load_saved_policies("TowardPOI{0}".format(skill_id), rover_id, srun)
            elif p["skill_type"] == "Target_Quadrant":
                weights = load_saved_policies("TowardQuadrant{0}".format(skill_id), rover_id, srun)
            rd.rovers[rov].get_weights(weights)
            rd.rovers[rov].reset_rover()
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 0] = rd.rovers[rov].x_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 1] = rd.rovers[rov].y_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 2] = rd.rovers[rov].theta_pos

        for rov in rd.rovers:  # Initial rover scan of environment
            rd.rovers[rov].scan_environment(rd.rovers, rd.pois)
        for poi in rd.pois:
            rd.pois[poi].update_observer_distances(rd.rovers)

        rewards = [[] for i in range(n_rovers)]
        for step_id in range(rover_steps):
            # Rover takes an action in the world
            for rov in rd.rovers:
                rd.rovers[rov].step(rd.world_x, rd.world_y)
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 0] = rd.rovers[rov].x_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 1] = rd.rovers[rov].y_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 2] = rd.rovers[rov].theta_pos

            # Rover scans environment and observer distances are updated
            for rk in rd.rovers:  # Rover scans environment
                rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
            for poi in rd.pois:
                rd.pois[poi].update_observer_distances(rd.rovers)

            # Calculate Global Reward
            for rover_id in range(n_rovers):
                if p["skill_type"] == "Target_POI":
                    rewards[rover_id].append(target_poi_reward(rover_id, rd.pois, skill_id))
                elif p["skill_type"] == "Target_Quadrant":
                    rewards[rover_id].append(target_quadrant_reward(rover_id, rd.pois, skill_id))

        for rover_id in range(n_rovers):
            skill_performance.append(sum(rewards[rover_id]))

        create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths")
        create_csv_file(skill_performance, "Output_Data/", "Skill{0}_Performance.csv".format(skill_id))

        srun += 1


def test_custom_skills(skill_id):
    """
    Test suggestions using the pre-trained policy bank
    """
    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    srun = p["starting_srun"]
    while srun < stat_runs:
        skill_performance = []  # Keep track of team performance throughout training

        # Reset rover and record initial position
        for rov in rd.rovers:
            rd.rovers[rov].reset_rover()
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 0] = rd.rovers[rov].x_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 1] = rd.rovers[rov].y_pos
            final_rover_path[srun, rd.rovers[rov].self_id, 0, 2] = rd.rovers[rov].theta_pos

        for rov in rd.rovers:  # Initial rover scan of environment
            rd.rovers[rov].scan_environment(rd.rovers, rd.pois)
        for poi in rd.pois:
            rd.pois[poi].update_observer_distances(rd.rovers)

        rewards = [[] for i in range(n_rovers)]
        for step_id in range(rover_steps):
            # Rover takes an action in the world
            for rov in rd.rovers:
                rd.rovers[rov].rover_actions = travel_to_poi(skill_id, rd.pois, rd.rovers[rov].x_pos, rd.rovers[rov].y_pos)
                rd.rovers[rov].custom_step(rd.world_x, rd.world_y)
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 0] = rd.rovers[rov].x_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 1] = rd.rovers[rov].y_pos
                final_rover_path[srun, rd.rovers[rov].self_id, step_id + 1, 2] = rd.rovers[rov].theta_pos

            # Rover scans environment and observer distances are updated
            for rk in rd.rovers:  # Rover scans environment
                rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
            for poi in rd.pois:
                rd.pois[poi].update_observer_distances(rd.rovers)

            # Calculate Global Reward
            for rover_id in range(n_rovers):
                if p["skill_type"] == "Target_POI":
                    rewards[rover_id].append(target_poi_reward(rover_id, rd.pois, skill_id))
                elif p["skill_type"] == "Target_Quadrant":
                    rewards[rover_id].append(target_quadrant_reward(rover_id, rd.pois, skill_id))

        for rover_id in range(n_rovers):
            skill_performance.append(sum(rewards[rover_id]))

        create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths")
        create_csv_file(skill_performance, "Output_Data/", "Skill{0}_Performance.csv".format(skill_id))

        srun += 1


if __name__ == '__main__':
    # Test Performance of Skills in Agent Skill Set
    if p["custom_skills"]:
        for skill_id in range(p["n_skills"]):
            print("Testing Skill: ", skill_id)
            test_custom_skills(skill_id)
            if p["vis_running"]:
                run_visualizer()
    else:
        for skill_id in range(p["n_skills"]):
            print("Testing Skill: ", skill_id)
            test_trained_skills(skill_id)
            if p["vis_running"]:
                run_visualizer()
