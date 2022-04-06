parameters = {}

# Run Parameters
parameters["stat_runs"] = 1
parameters["starting_srun"] = 0
parameters["generations"] = 10  # Number of generations for CCEA in each stat run
parameters["pbank_generations"] = 10  # Number of generations used for training policy bank
parameters["reward_type"] = "Difference"  # Global, Difference, DPP, SDPP (for non-suggestion training)
parameters["coupling"] = 1  # Number of rovers required to view a POI for credit
parameters["sample_rate"] = 20  # Spacing for collecting performance data during training (every X generations)

# Visualizer
parameters["vis_running"] = False  # True keeps visualizer from closing until you 'X' out of window

# Domain parameters
parameters["x_dim"] = 50.0  # X-Dimension of the rover map
parameters["y_dim"] = 50.0  # Y-Dimension of the rover map
parameters["n_rovers"] = 3  # Number of rovers on map
parameters["n_poi"] = 2  # Number of POIs on map
parameters["steps"] = 30  # Number of time steps rovers take each episode
parameters["poi_config_type"] = "Two_POI"  # Random, Two_POI, Four_Corners, Circle, Con_Circle, Four_Quadrants
parameters["rover_config_type"] = "Concentrated"  # Random, Concentrated, Four_Quadrants
parameters["active_hazards"] = False  # Determine if hazard zones are active (True) or inactive (False)

# Rover Parameters
parameters["sensor_model"] = "summed"  # Should either be "density" or "summed"
parameters["angle_res"] = 360 / 8  # Resolution of sensors (determines number of sectors)
parameters["observation_radius"] = 4.0  # Maximum range at which rovers can observe a POI
parameters["min_distance"] = 1.0  # Minimum distance which may appear in the denominator of credit eval functions
parameters["dmax"] = 2.5  # Maximum distance a rover can move in a single time step

# Suggestion Parameters
parameters["policy_bank_type"] = "Target_POI"  # Target_Quadrant or Target_POI
parameters["n_skills"] = 2  # Number of pre-trained policies in the policy bank
parameters["suggestion_type"] = "Custom"  # Identical, Unique, Random, Custom
parameters["s_inputs"] = int(2 * (360 / parameters["angle_res"]))
parameters["s_hidden"] = 12
parameters["s_outputs"] = parameters["n_skills"]

# Neural network parameters for rover motor control
parameters["n_inputs"] = int(2 * (360 / parameters["angle_res"]))
parameters["n_hidden"] = 10
parameters["n_outputs"] = 2

# CCEA parameters
parameters["pop_size"] = 40
parameters["mutation_chance"] = 0.1  # Probability that a mutation will occur
parameters["mutation_rate"] = 0.2  # How much a weight is allowed to change
parameters["epsilon"] = 0.1  # For e-greedy selection in CCEA
parameters["n_elites"] = 1  # How many elites to carry over during selection
