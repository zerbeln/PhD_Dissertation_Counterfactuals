parameters = {}

# Run Parameters
parameters["stat_runs"] = 1
parameters["generations"] = 1000  # Number of generations for CCEA in each stat run
parameters["new_world_config"] = 1  # 0 = False -> Reuse existing world config, 1 = True -> Use new world config

# Visualizer
parameters["running"] = False  # True keeps visualizer from closing until you 'X' out of window

# Domain parameters
parameters["n_rovers"] = 3  # Number of rovers on map
parameters["coupling"] = 1  # Number of rovers required to view a POI for credit
parameters["n_poi"] = 3  # Number of POIs on map
parameters["n_steps"] = 30  # Number of steps rovers take each episode
parameters["min_distance"] = 1.0  # Minimum distance which may appear in the denominator of credit eval functions
parameters["x_dim"] = 40.0  # X-Dimension of the rover map
parameters["y_dim"] = 40.0  # Y-Dimension of the rover map
parameters["min_obs_dist"] = 3.0  # Minimum distance rovers must be to observe POIs
parameters["angle_res"] = 90  # Resolution of sensors (determines number of sectors)
parameters["sensor_model"] = "summed"  # Should either be "density" or "closest" or "summed"

# Neural network parameters
parameters["n_inputs"] = 8
parameters["n_hnodes"] = 9
parameters["n_outputs"] = 2
parameters["mem_block_size"] = 9

# CCEA parameters
parameters["pop_size"] = 40
parameters["mut_prob"] = 0.1  # Probability that a mutation will occur
parameters["mut_rate"] = 0.05  # How much a weight is allowed to change
parameters["epsilon"] = 0.1  # For e-greedy selection in CCEA
parameters["n_elites"] = 3
