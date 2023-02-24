from EvolutionaryAlgorithms.ccea import CCEA
from CKI.cki_rewards import *
from RoverDomainCore.rover_domain import RoverDomain
import os
import numpy as np
import csv
from parameters import parameters as p
import random
from global_functions import save_best_policies


def save_time_history(time_history, file_name):
    """
    Save reward data as a CSV file for graph generation. CSV is appended each time function is called.
    """

    dir_name = 'Output_Data/'  # Intended directory for output files
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, file_name)
    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(time_history)


def save_skill_reward_history(rover_id, reward_history, file_name):
    """
    Save reward data as a CSV file for graph generation. CSV is appended each time function is called.
    """

    dir_name = 'Output_Data/Rover{0}'.format(rover_id)  # Intended directory for output files
    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    save_file_name = os.path.join(dir_name, file_name)
    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(reward_history)


def train_towards_teammates():
    """
    Train rover policy for travelling towards teammates
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["pbank_generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = CCEA(n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for pkey in pops:
            pops[pkey].create_new_population()

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()
                    rover_id = rd.rovers[rk].self_id
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rd.rovers[rk].get_weights(weights)
                for rk in rd.rovers:  # Initial rover scan of environment
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                        reward = towards_teammates_reward(rd.rovers, rd.rovers[rk].self_id)
                        rover_rewards[rd.rovers[rk].self_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    pops["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "TowardTeammates", rover_id)


def train_away_teammates():
    """
    Train rover policy for travelling away from teammates
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["pbank_generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for pkey in pops:
            pops[pkey].create_new_population()

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()
                    rover_id = rd.rovers[rk].self_id
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rd.rovers[rk].get_weights(weights)
                for rk in rd.rovers:  # Initial rover scan of environment
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                        reward = away_teammates_reward(rd.rovers, rd.rovers[rk].self_id)
                        rover_rewards[rd.rovers[rk].self_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    pops["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "AwayTeammates", rover_id)


def train_towards_poi():
    """
    Train rover policy for travelling towards POI
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["pbank_generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for pkey in pops:
            pops[pkey].create_new_population()

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    rd.rovers[rk].reset_rover()
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rd.rovers[rk].get_weights(weights)
                for rk in rd.rovers:  # Initial rover scan of environment
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)
                    for rover_id in range(n_rovers):
                        reward = towards_poi_reward(rover_id, rd.pois)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    pops["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "TowardPOI", rover_id)


def train_away_poi():
    """
    Train rover policy for travelling away from POI
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["pbank_generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for pkey in pops:
            pops[pkey].create_new_population()

        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    rd.rovers[rk].reset_rover()
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rd.rovers[rk].get_weights(weights)
                for rk in rd.rovers:  # Initial rover scan of environment
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)
                    for rover_id in range(n_rovers):
                        reward = away_poi_reward(rover_id, rd.pois)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    pops["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "AwayPOI", rover_id)


def train_target_poi(rover_targets):
    """
    Train rover skills for travelling towards specific POI
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["pbank_generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    # Perform statistical runs
    srun = p["starting_srun"]
    while srun < stat_runs:
        print("Run: %i" % srun)

        # Reset Rover and CCEA Pop
        for pkey in pops:
            pops[pkey].create_new_population()

        skill_rewards = [[] for i in range(n_rovers)]
        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()

            # Each policy in CCEA is tested in randomly selected teams
            for team_id in range(population_size):
                rover_rewards = np.zeros((n_rovers, rover_steps))  # Keep track of rover rewards at each t
                for rk in rd.rovers:
                    rd.rovers[rk].reset_rover()
                    pol_id = int(pops["EA{0}".format(rd.rovers[rk].self_id)].team_selection[team_id])
                    weights = pops["EA{0}".format(rd.rovers[rk].self_id)].population["pol{0}".format(pol_id)]
                    rd.rovers[rk].get_weights(weights)

                # Initial rover scan of environment
                for rk in rd.rovers:
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                # Rover processes scan information and acts in current time step
                for step_id in range(rover_steps):
                    for rk in rd.rovers:
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)
                    for rover_id in range(n_rovers):
                        reward = target_poi_reward(rover_id, rd.pois, rover_targets[rover_id])
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    pops["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            # Track Skill Training Performance
            if gen % sample_rate == 0 or gen == generations-1:
                for rover_id in range(n_rovers):
                    skill_rewards[rover_id].append(max(pops["EA{0}".format(rover_id)].fitness))

            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            target_poi = rover_targets[rover_id]
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "TowardPOI{0}".format(target_poi), rover_id)
            save_skill_reward_history(rover_id, skill_rewards[rover_id], "Skill{0}_Training.csv".format(target_poi))

        srun += 1


def train_target_quadrant(target_q):
    """
    Train rover policies for travelling towards POI within a specific quadrant
    """
    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["pbank_generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    sample_rate = p["sample_rate"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # World Setup
    rd = RoverDomain()
    rd.load_world()

    # Create dictionary of rover instances
    pops = {}
    for rover_id in range(n_rovers):
        pops["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=n_inp, n_hid=n_hid, n_out=n_out)

    srun = p["starting_srun"]
    while srun < stat_runs:  # Perform statistical runs
        print("Run: %i" % srun)

        # Create new CCEA populations
        for pkey in pops:
            pops[pkey].create_new_population()

        skill_rewards = [[] for i in range(n_rovers)]
        for gen in range(generations):
            for pkey in pops:
                pops[pkey].select_policy_teams()
            for team_id in range(population_size):  # Each policy in CCEA is tested in teams
                rover_rewards = np.zeros((n_rovers, rover_steps))
                for rk in rd.rovers:
                    rover_id = rd.rovers[rk].self_id
                    rd.rovers[rk].reset_rover()
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(pol_id)]
                    rd.rovers[rk].get_weights(weights)

                # Initial rover scan of environment
                for rk in rd.rovers:
                    rd.rovers[rk].scan_environment(rd.rovers, rd.pois)

                for step_id in range(rover_steps):
                    for rk in rd.rovers:  # Rover processes scan information and acts
                        rd.rovers[rk].step(rd.world_x, rd.world_y)
                    for rk in rd.rovers:  # Rover scans environment
                        rd.rovers[rk].scan_environment(rd.rovers, rd.pois)
                    for poi in rd.pois:
                        rd.pois[poi].update_observer_distances(rd.rovers)
                    for rover_id in range(n_rovers):
                        reward = target_quadrant_reward(rover_id, rd.pois, target_q)
                        rover_rewards[rover_id, step_id] = reward

                # Update fitness of policies using reward information
                for rover_id in range(n_rovers):
                    pol_id = int(pops["EA{0}".format(rover_id)].team_selection[team_id])
                    pops["EA{0}".format(rover_id)].fitness[pol_id] = sum(rover_rewards[rover_id])

            # Track Skill Training Performance
            if gen % sample_rate == 0 or gen == generations - 1:
                for rover_id in range(n_rovers):
                    skill_rewards[rover_id].append(max(pops["EA{0}".format(rover_id)].fitness))
            for pkey in pops:
                pops[pkey].down_select()  # Choose new parents and create new offspring population

        # Record best policy trained for each rover
        for rover_id in range(n_rovers):
            policy_id = np.argmax(pops["EA{0}".format(rover_id)].fitness)
            weights = pops["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "TowardQuadrant{0}".format(target_q), rover_id)
            save_skill_reward_history(rover_id, skill_rewards[rover_id], "Skill{0}_Training.csv".format(target_q))

        srun += 1


if __name__ == '__main__':
    """
    Train policy playbooks for rover team
    """

    rover_skills = []
    if p["randomize_skills"]:
        for rover_id in range(p["n_rovers"]):
            skill_sample = random.sample(range(p["n_skills"]), p["n_skills"])
            rover_skills.append(skill_sample)
    else:
        for rover_id in range(p["n_rovers"]):
            rover_skills.append([])
            for skill_id in range(p["n_skills"]):
                rover_skills[rover_id].append(skill_id)

    print(rover_skills)

    if p["skill_type"] == "Target_POI":
        for skill_id in range(p["n_poi"]):
            print("Training Skill: ", skill_id)
            target_skills = []
            for rover_id in range(p["n_rovers"]):
                target_skills.append(rover_skills[rover_id][skill_id])
            train_target_poi(target_skills)
    elif p["skill_type"] == "Target_Quadrant":
        for skill_id in range(4):
            print("Training Skill: ", skill_id)
            target_skills = []
            for rover_id in range(p["n_rovers"]):
                target_skills.append(rover_skills[rover_id][skill_id])
            train_target_quadrant(target_skills)
    else:
        print("INCORRECT SKILL TRAINING METHOD")
