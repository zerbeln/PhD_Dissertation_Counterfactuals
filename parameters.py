"""
This file contains all constant test parameters which may be altered from this single
location for convenience.
"""

class Parameters:

    # Run Parameters
    stat_runs = 1
    generations = 100  # Number of generations for CCEA in each stat run

    # Domain parameters
    rover_types = 'homogeneous'  # Use 'homogeneous' for uniform rovers, and 'heterogeneous' for non-uniform rovers
    reward_type = 0  # 0 for global, 1 for difference, 2 for d++, 3 for s-d++
    num_rovers = 12  # Number of rovers on map (GETS MULTIPLIED BY NUMBER OF TYPES)
    num_types = 1  # How many types of rovers are on the map
    coupling = 3  # Number of rovers required to view a POI for credit
    num_pois = 10  # Number of POIs on map
    num_steps = 30  # Number of steps rovers take each episode
    min_distance = 1.0  # Minimum distance which may appear in the denominator of credit eval functions
    world_size = 30
    activation_dist = 4.0  # Minimum distance rovers must be to observe POIs
    angle_resolution = 90  # Resolution of sensors (determines number of sectors)
    sensor_model = "closest"  # Should either be density or closest (Use closest for evolutionary domain)

    # Neural network parameters
    num_inputs = 8
    num_nodes = 9
    num_outputs = 2

    # CCEA parameters
    mutation_rate = 0.1
    epsilon = 0.1
    pop_size = 10
