"""
This file contains all constant test parameters which may be altered from this single
location for convenience.
"""


class Parameters:

    """
    TODO: Heterogeneous rover domain code currently not implemented
    """

    # Run Parameters
    stat_runs = 1
    generations = 1  # Number of generations for CCEA in each stat run
    visualizer_on = True  # Turn visualizer on or off (TURN OFF FOR MULTIPLE STAT RUNS)

    # Domain parameters
    team_types = 'homogeneous'  # Switch between 'homogeneous' and 'heterogeneous' rover domains
    num_rovers = 12  # Number of rovers on map (GETS MULTIPLIED BY NUMBER OF TYPES)
    coupling = 3  # Number of rovers required to view a POI for credit
    num_pois = 10  # Number of POIs on map
    num_steps = 30  # Number of steps rovers take each episode
    min_distance = 0.5  # Minimum distance which may appear in the denominator of credit eval functions
    x_dim = 30  # X-Dimension of the rover map
    y_dim = 30  # Y-Dimension of the rover map
    min_observation_dist = 4.0  # Minimum distance rovers must be to observe POIs
    angle_resolution = 90  # Resolution of sensors (determines number of sectors)
    sensor_model = "closest"  # Should either be "density" or "closest"

    # Neural network parameters
    num_inputs = 8
    num_nodes = 9
    num_outputs = 2

    # CCEA parameters
    mutation_rate = 0.1
    epsilon = 0.1
    pop_size = 15

    # User specific parameters
    reward_type = "DPP"  # Switch between reward functions
