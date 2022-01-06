"""
This contains fragments of depricated code that are no longer used, but are being kept in case they may be useful
in the future.
"""

def train_suggestions_loose_direct():
    """
    Train suggestions in the loosely coupled rover domain using direct action output (no playbook)
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    assert(p["coupling"] == 1)

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # Suggestion Parameters
    n_suggestions = p["n_suggestions"]
    s_inp = p["s_inputs"]
    s_hid = p["s_hidden"]
    s_out = p["s_outputs"]

    rd = RoverDomain()  # Create instance of the rover domain

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=s_inp, n_out=s_out, n_hid=s_hid)
        rovers["SN{0}".format(rover_id)] = SuggestionNetwork(s_inp, s_out, s_hid)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Load World Configuration
        rd.load_world(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population

        # Create list of suggestions for rovers to use during training
        rover_suggestions = []
        for rover_id in range(n_rovers):
            if rover_id % 2 == 0:
                s_type = [1, 0]
            else:
                s_type = [0, 1]
            rover_suggestions.append(s_type)

        s_id = np.zeros(n_rovers, int)  # Identifies what suggestion each rover is using
        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()

            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                # Get weights for suggestion interpreter
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["SN{0}".format(rover_id)].get_weights(weights)

                for sgst in range(n_suggestions):
                    for rover_id in range(n_rovers):
                        rovers["Rover{0}".format(rover_id)].reset_rover()
                        s_id[rover_id] = int(rover_suggestions[rover_id][sgst])

                    for rover_id in range(n_rovers):  # Initial rover scan of environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, s_id[rover_id])
                        sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                        sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                        rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                        # Determine action based on sensor inputs and suggestion
                        sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        rovers["Rover{0}".format(rover_id)].rover_actions = sug_outputs.copy()

                    rover_rewards = np.zeros((n_rovers, rover_steps))  # Keep track of reward earned by each rover at t
                    for step_id in range(rover_steps):
                        # Rover takes an action in the world
                        for rover_id in range(n_rovers):
                            rovers["Rover{0}".format(rover_id)].suggestion_step(rd.world_x, rd.world_y)

                        # Rover scans environment and processes suggestions
                        for rover_id in range(n_rovers):
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                            rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                            suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, s_id[rover_id])
                            sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                            sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                            rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                            # Choose policy based on sensors and suggestion
                            sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                            rovers["Rover{0}".format(rover_id)].rover_actions = sug_outputs.copy()

                        g_reward = rd.calc_global_loose()
                        dif_reward = calc_sd_reward(rd.observer_distances, rd.pois, g_reward, s_id)
                        for rover_id in range(n_rovers):
                            rover_rewards[rover_id, step_id] = dif_reward[rover_id]

                    for rover_id in range(n_rovers):
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[policy_id] += sum(rover_rewards[rover_id])

                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] /= n_suggestions

            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()

        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)


def train_suggestions_tight_direct():
    """
    Train suggestions in the tightly coupled rover domain using direct action output (no playbook)
    """

    # Parameters
    stat_runs = p["stat_runs"]
    generations = p["generations"]
    population_size = p["pop_size"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    assert(p["coupling"] > 1)

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # Suggestion Parameters
    n_suggestions = p["n_suggestions"]
    s_inp = p["s_inputs"]
    s_hid = p["s_hidden"]
    s_out = p["s_outputs"]

    rd = RoverDomain()  # Create instance of the rover domain

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["EA{0}".format(rover_id)] = Ccea(population_size, n_inp=s_inp, n_out=s_out, n_hid=s_hid)
        rovers["SN{0}".format(rover_id)] = SuggestionNetwork(s_inp, s_out, s_hid)

    for srun in range(stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)

        # Load World Configuration
        rd.load_world(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            rovers["EA{0}".format(rover_id)].create_new_population()  # Create new CCEA population

        s_id = np.zeros(n_rovers, int)  # Identifies what suggestion each rover is using
        # Create list of suggestions for rovers to use during training
        rover_suggestions = []
        for rover_id in range(n_rovers):
            s_type = [0, 1]
            rover_suggestions.append(s_type)

        for gen in range(generations):
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].select_policy_teams()
                rovers["EA{0}".format(rover_id)].reset_fitness()

            for team_number in range(population_size):  # Each policy in CCEA is tested in teams
                # Get weights for suggestion interpreter
                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
                    rovers["SN{0}".format(rover_id)].get_weights(weights)

                for sgst in range(n_suggestions):
                    for rover_id in range(n_rovers):
                        rovers["Rover{0}".format(rover_id)].reset_rover()
                        s_id[rover_id] = int(rover_suggestions[rover_id][sgst])

                    for rover_id in range(n_rovers):  # Initial rover scan of environment
                        rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                        suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, s_id[rover_id])
                        sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                        sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                        rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                        # Determine action based on sensor inputs and suggestion
                        sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                        rovers["Rover{0}".format(rover_id)].rover_actions = sug_outputs.copy()

                    rover_rewards = np.zeros((n_rovers, rover_steps))  # Keep track of reward earned by each rover at t
                    for step_id in range(rover_steps):
                        # Rover takes an action in the world
                        for rover_id in range(n_rovers):
                            rovers["Rover{0}".format(rover_id)].suggestion_step(rd.world_x, rd.world_y)

                        # Rover scans environment and processes suggestions
                        for rover_id in range(n_rovers):
                            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                            rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                            suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, s_id[rover_id])
                            sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                            sug_input = np.concatenate((suggestion, sensor_data), axis=0)
                            rovers["SN{0}".format(rover_id)].get_inputs(sug_input)

                            # Choose policy based on sensors and suggestion
                            sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                            rovers["Rover{0}".format(rover_id)].rover_actions = sug_outputs.copy()

                        g_reward = rd.calc_global_tight()
                        dpp_reward = calc_sdpp(rd.observer_distances, rd.pois, g_reward, s_id)
                        for rover_id in range(n_rovers):
                            rover_rewards[rover_id, step_id] = dpp_reward[rover_id]

                    for rover_id in range(n_rovers):
                        policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                        rovers["EA{0}".format(rover_id)].fitness[policy_id] += sum(rover_rewards[rover_id])

                for rover_id in range(n_rovers):
                    policy_id = int(rovers["EA{0}".format(rover_id)].team_selection[team_number])
                    rovers["EA{0}".format(rover_id)].fitness[policy_id] /= n_suggestions

            # Choose new parents and create new offspring population
            for rover_id in range(n_rovers):
                rovers["EA{0}".format(rover_id)].down_select()

        for rover_id in range(n_rovers):
            policy_id = np.argmax(rovers["EA{0}".format(rover_id)].fitness)
            weights = rovers["EA{0}".format(rover_id)].population["pol{0}".format(policy_id)]
            save_best_policies(weights, srun, "SelectionWeights{0}".format(rover_id), rover_id)

def test_suggestions_direct(sgst):
    """
    Test suggestions using direct action output (no policy bank)
    """
    # Parameters
    stat_runs = p["stat_runs"]
    n_rovers = p["n_rovers"]
    rover_steps = p["steps"]
    domain_type = p["domain_type"]

    # Rover Motor Control
    n_inp = p["n_inputs"]
    n_hid = p["n_hidden"]
    n_out = p["n_outputs"]

    # Suggestion Parameters
    s_inp = p["s_inputs"]
    s_hid = p["s_hidden"]
    s_out = p["s_outputs"]

    rd = RoverDomain()  # Create instance of the rover domain

    # Create dictionary for each instance of rover and corresponding NN and EA population
    rovers = {}
    for rover_id in range(n_rovers):
        rovers["Rover{0}".format(rover_id)] = Rover(rover_id, n_inp=n_inp, n_hid=n_hid, n_out=n_out)
        rovers["SN{0}".format(rover_id)] = SuggestionNetwork(s_inp, s_out, s_hid)

    average_reward = 0
    reward_history = []
    final_rover_path = np.zeros((stat_runs, n_rovers, rover_steps + 1, 3))
    for srun in range(stat_runs):  # Perform statistical runs

        # Load Trained Suggestion Interpreter Weights
        for rover_id in range(n_rovers):
            s_weights = load_saved_policies('SelectionWeights{0}'.format(rover_id), rover_id, srun)
            rovers["SN{0}".format(rover_id)].get_weights(s_weights)

        # Load World Configuration
        rd.load_world(srun)
        for rover_id in range(n_rovers):
            rovers["Rover{0}".format(rover_id)].initialize_rover(srun)
            final_rover_path[srun, rover_id, 0, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
            final_rover_path[srun, rover_id, 0, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
            final_rover_path[srun, rover_id, 0, 2] = rovers["Rover{0}".format(rover_id)].pos[2]

        reward_history = []  # Keep track of team performance throughout training
        for rover_id in range(n_rovers):  # Initial rover scan of environment
            suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, sgst)
            rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
            sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
            sug_input = np.concatenate((suggestion, sensor_data), axis=0)
            rovers["SN{0}".format(rover_id)].get_inputs(sug_input)
            sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
            rovers["Rover{0}".format(rover_id)].rover_actions = sug_outputs.copy()

        g_rewards = np.zeros(rover_steps)
        for step_id in range(rover_steps):
            # Rover takes an action in the world
            for rover_id in range(n_rovers):
                rovers["Rover{0}".format(rover_id)].suggestion_step(rd.world_x, rd.world_y)
                final_rover_path[srun, rover_id, step_id + 1, 0] = rovers["Rover{0}".format(rover_id)].pos[0]
                final_rover_path[srun, rover_id, step_id + 1, 1] = rovers["Rover{0}".format(rover_id)].pos[1]
                final_rover_path[srun, rover_id, step_id + 1, 2] = rovers["Rover{0}".format(rover_id)].pos[2]

            # Rover scans environment and processes suggestions
            for rover_id in range(n_rovers):
                suggestion = construct_counterfactual_state(rd.pois, rovers, rover_id, sgst)
                rovers["Rover{0}".format(rover_id)].scan_environment(rovers, rd.pois, n_rovers)
                rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                sensor_data = rovers["Rover{0}".format(rover_id)].sensor_readings
                sug_inputs = np.concatenate((suggestion, sensor_data), axis=0)
                rovers["SN{0}".format(rover_id)].get_inputs(sug_inputs)
                rd.update_observer_distances(rover_id, rovers["Rover{0}".format(rover_id)].poi_distances)
                sug_outputs = rovers["SN{0}".format(rover_id)].get_outputs()
                rovers["Rover{0}".format(rover_id)].rover_actions = sug_outputs.copy()

            # Calculate Global Reward
            if domain_type == "Loose":
                g_rewards[step_id] = rd.calc_global_loose()
            else:
                g_rewards[step_id] = rd.calc_global_tight()

        reward_history.append(sum(g_rewards))
        average_reward += sum(g_rewards)

        save_rover_path(final_rover_path, "Rover_Paths")
    save_reward_history(reward_history, "Final_GlobalRewards.csv")
    average_reward /= stat_runs
    print(average_reward)
    run_visualizer()

