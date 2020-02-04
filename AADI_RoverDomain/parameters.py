import yaml


class Parameters:

    def __init__(self):
        # Run Parameters
        self.stat_runs = 1
        self.generations = 1  # Number of generations for CCEA in each stat run
        self.new_world_config = 0  # 0 = False -> Reuse existing world config, 1 = True -> Use new world config

        # Visualizer
        self.running = False  # True keeps visualizer from closing until you 'X' out of window

        # Domain parameters
        self.num_rovers = 3  # Number of rovers on map
        self.coupling = 1  # Number of rovers required to view a POI for credit
        self.num_pois = 3  # Number of POIs on map
        self.num_steps = 30  # Number of steps rovers take each episode
        self.min_distance = 1.0  # Minimum distance which may appear in the denominator of credit eval functions
        self.x_dim = 30.0  # X-Dimension of the rover map
        self.y_dim = 30.0  # Y-Dimension of the rover map
        self.min_observation_dist = 3.0  # Minimum distance rovers must be to observe POIs
        self.angle_resolution = 90  # Resolution of sensors (determines number of sectors)
        self.sensor_model = "summed"  # Should either be "density" or "closest" or "summed"

        # Neural network parameters
        self.num_inputs = 8
        self.num_nodes = 9
        self.num_outputs = 2

        # CCEA parameters
        self.mutation_prob = 0.1  # Probability that a mutation will occur
        self.mutation_rate = 0.05  # How much a weight is allowed to change
        self.epsilon = 0.1  # For e-greedy selection in CCEA
        self.parent_pop_size = 30
        self.offspring_pop_size = 30
        self.total_pop_size = self.parent_pop_size + self.offspring_pop_size

    def load_yaml(self, filename):
        """
        loads a set of parameters into this object with setattr
        Does not require that all parameters are set, i.e. you can optionally set a specific set of parameters
        :param filename:
        :return:
        """
        with open(file=filename) as f:
            params = yaml.load(f)
        for key in params:
            setattr(self, key, params[key])

    def save_yaml(self, filename):
        """
        Saves ALL the parameters into a yaml file, not just the optionally set ones
        :param filename:
        :return:
        """
        with open(file=filename, mode='w') as f:
            yaml.dump(vars(self), f)


if __name__ == '__main__':
    with open("/tmp/param_tests.yaml", 'w') as f:
        f.write("""
            test_a: 10
            stat_runs: 100
            """)
    parameters = Parameters()
    print("Before load")
    print(parameters.stat_runs)
    parameters.load_yaml("/tmp/param_tests.yaml")
    print("After load")
    print(parameters.test_a)
    print(parameters.stat_runs)
    parameters.save_yaml("/tmp/param_tests_out.yaml")
