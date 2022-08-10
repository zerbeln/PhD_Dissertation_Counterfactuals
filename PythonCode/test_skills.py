from RoverDomain_Core.rover_domain import RoverDomain
from Visualizer.visualizer import run_visualizer
import numpy as np
from parameters import parameters as p
from CBA.custom_rover_skills import get_custom_action
from global_functions import create_csv_file, create_pickle_file, load_saved_policies
from CBA.cba import calculate_poi_sectors


def test_custom_skills(skill_id):
    """
    Test suggestions using the pre-trained policy bank
    """
    # World Setup
    rd = RoverDomain()
    rd.load_world()
    calculate_poi_sectors(rd.pois)

    final_rover_path = np.zeros((p["stat_runs"], p["n_rovers"], p["steps"], 3))
    srun = p["starting_srun"]
    skill_performance = []  # Keep track of team performance throughout training
    while srun < p["stat_runs"]:
        # Reset rover and record initial position
        rd.reset_world()
        poi_rewards = np.zeros((p["n_poi"], p["steps"]))
        for step_id in range(p["steps"]):
            # Rover takes an action in the world
            rover_actions = []
            for rv in rd.rovers:
                action = get_custom_action(skill_id, rd.pois, rd.rovers[rv].loc[0], rd.rovers[rv].loc[1])
                rover_actions.append(action)
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 0] = rd.rovers[rv].loc[0]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 1] = rd.rovers[rv].loc[1]
                final_rover_path[srun, rd.rovers[rv].rover_id, step_id, 2] = rd.rovers[rv].loc[2]

            # Environment takes in rover actions and returns next state and global reward
            step_rewards = rd.step(rover_actions)
            for poi_id in range(p["n_poi"]):
                poi_rewards[poi_id, step_id] = step_rewards[poi_id]

        # Calculate episodic global reward
        g_reward = 0
        for p_reward in poi_rewards:
            g_reward += max(p_reward)
        skill_performance.append(g_reward)
        srun += 1

    create_pickle_file(final_rover_path, "Output_Data/", "Rover_Paths")
    create_csv_file(skill_performance, "Output_Data/", "Skill{0}_Performance.csv".format(skill_id))


if __name__ == '__main__':
    # Test Performance of Skills in Agent Skill Set
    skill_id = 2
    print("Testing Skill: ", skill_id)
    test_custom_skills(skill_id)
    if p["vis_running"]:
        run_visualizer()
