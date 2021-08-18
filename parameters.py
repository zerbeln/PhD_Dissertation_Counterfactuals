parameters = {}

# Run Parameters
parameters["stat_runs"] = 1
parameters["generations"] = 500  # Number of generations for CCEA in each stat run
parameters["test_type"] = "Train_Pol_Select"  # Create_Bank, Create_World, Train_Pol_Select, Test, Full_Train, Standard
parameters["reward_type"] = "Global"  # Global, Difference, DPP, SDPP
parameters["g_type"] = "Loose"  # Loose or Tight
parameters["sample_rate"] = 20  # Spacing for collecting performance data during training (every X generations)

# Visualizer
parameters["vis_running"] = False  # True keeps visualizer from closing until you 'X' out of window

# Domain parameters
parameters["x_dim"] = 50.0  # X-Dimension of the rover map
parameters["y_dim"] = 50.0  # Y-Dimension of the rover map
parameters["n_rovers"] = 10  # Number of rovers on map
parameters["coupling"] = 1  # Number of rovers required to view a POI for credit
parameters["n_poi"] = 10  # Number of POIs on map
parameters["steps"] = 20  # Number of time steps rovers take each episode

# Suggestion Parameters
parameters["n_suggestions"] = 4  # Number of suggestions a rover should learn
parameters["suggestion"] = 1
parameters["s_inputs"] = 16
parameters["s_hidden"] = 12
parameters["s_outputs"] = 2

# Rover Parameters
parameters["sensor_model"] = "summed"  # Should either be "density" or "summed"
parameters["angle_res"] = 90  # Resolution of sensors (determines number of sectors)
parameters["observation_radius"] = 4.0  # Maximum range at which rovers can observe a POI
parameters["min_distance"] = 1.0  # Minimum distance which may appear in the denominator of credit eval functions
parameters["dmax"] = 3.0  # Maximum distance a rover can move in a single time step

# Neural network parameters for rover motor control
parameters["n_inputs"] = 8
parameters["n_hidden"] = 10
parameters["n_outputs"] = 2

# CCEA parameters
parameters["pop_size"] = 20
parameters["mutation_chance"] = 0.1  # Probability that a mutation will occur
parameters["mutation_rate"] = 0.1  # How much a weight is allowed to change
parameters["epsilon"] = 0.1  # For e-greedy selection in CCEA
parameters["n_elites"] = 1  # How many elites to carry over during selection
