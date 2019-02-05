"""
This file contains all constant test parameters which may be altered from this single
location for convenience.
"""

class Parameters:

    # Run Parameters
    stat_runs = 10
    generations = 1000  # Number of generations for CCEA in each stat run
    tests_per_gen = 1  # Number of tests run after each generation

    # Domain parameters
    num_rovers = 12  # Number of rovers on map
    num_pois = 10  # Number of POIs on map
    num_steps = 30  # Number of steps rovers take each episode


    min_distance = 1.0  # Minimum distance which may appear in the denominator of credit eval functions
    total_steps = 30  # Number of steps rovers take during each run of the world
    world_width = 30
    world_length = 30
    coupling = 3  # Number of rovers required to view a POI for credit
    activation_dist = 4.0  # Minimum distance rovers must be to observe POIs
    static_rovers = False  # False -> random initialization, True -> static initialization
    static_poi = False  # False -> random initialization, True -> static initialization

    # Neural network parameters
    num_inputs = 8
    num_nodes = 9
    num_outputs = 2

    # CCEA parameters
    mutation_rate = 0.1
    epsilon = 0.1
    population_size = 10
