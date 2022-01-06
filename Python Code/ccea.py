import numpy as np
import random
import copy
from parameters import parameters as p


class Ccea:
    def __init__(self, pop_size, n_inp=8, n_out=2, n_hid=10):
        self.population = {}
        self.pop_size = pop_size
        self.mut_rate = p["mutation_rate"]
        self.mut_chance = p["mutation_chance"]
        self.eps = p["epsilon"]
        self.fitness = np.zeros(self.pop_size)
        self.n_elites = p["n_elites"]  # Number of elites selected from each gen
        self.team_selection = np.ones(self.pop_size) * (-1)

        # Network Parameters that determine the number of weights to evolve
        self.n_inputs = n_inp
        self.n_outputs = n_out
        self.n_hidden = n_hid

    def reset_fitness(self):
        """
        Clear fitness vector of stored values
        """

        self.fitness = np.zeros(self.pop_size)

    def create_new_population(self):  # Re-initializes CCEA populations for new run
        """
        Create new populations (for beginning of stat run)
        """
        self.population = {}
        self.fitness = np.zeros(self.pop_size)

        for pol_id in range(self.pop_size):
            policy = {}
            policy["L1"] = np.random.standard_cauchy(self.n_inputs * self.n_hidden)
            policy["L2"] = np.random.standard_cauchy(self.n_hidden * self.n_outputs)
            policy["b1"] = np.random.standard_cauchy(self.n_hidden)
            policy["b2"] = np.random.standard_cauchy(self.n_outputs)

            self.population["pol{0}".format(pol_id)] = policy.copy()

    def select_policy_teams(self):
        """
        Determines which individuals from each population get paired together
        """

        self.team_selection = random.sample(range(self.pop_size), self.pop_size)

    def weight_mutate(self):
        """
        Mutate offspring populations (each weight has a probability of mutation)
        """

        pol_id = int(self.n_elites)  # Make sure that elites are not mutated
        while pol_id < self.pop_size:

            # First Weight Layer
            for w in self.population["pol{0}".format(pol_id)]["L1"]:
                rnum1 = random.uniform(0, 1)
                if rnum1 <= self.mut_chance:
                    w += np.random.normal(0, self.mut_rate) * w

            # Second Weight Layer
            for w in self.population["pol{0}".format(pol_id)]["L2"]:
                rnum2 = random.uniform(0, 1)
                if rnum2 <= self.mut_chance:
                    w += np.random.normal(0, self.mut_rate) * w

            # Output bias weights
            for w in self.population["pol{0}".format(pol_id)]["b1"]:
                rnum3 = random.uniform(0, 1)
                if rnum3 <= self.mut_chance:
                    w += np.random.normal(0, self.mut_rate) * w

            # Output layer weights
            for w in self.population["pol{0}".format(pol_id)]["b2"]:
                rnum4 = random.uniform(0, 1)
                if rnum4 <= self.mut_chance:
                    w += (np.random.normal(0, self.mut_rate)) * w

            pol_id += 1

    def binary_tournament_selection(self):
        """
        Select parents using binary tournament selection
        """
        new_population = {}
        for pol_id in range(self.pop_size):
            if pol_id < self.n_elites:
                new_population["pol{0}".format(pol_id)] = copy.deepcopy(self.population["pol{0}".format(pol_id)])
            else:
                p1 = random.randint(0, self.pop_size-1)
                p2 = random.randint(0, self.pop_size-1)
                while p1 == p2:
                    p2 = random.randint(0, self.pop_size - 1)

                if self.fitness[p1] > self.fitness[p2]:
                    new_population["pol{0}".format(pol_id)] = copy.deepcopy(self.population["pol{0}".format(p1)])
                elif self.fitness[p1] < self.fitness[p2]:
                    new_population["pol{0}".format(pol_id)] = copy.deepcopy(self.population["pol{0}".format(p2)])
                else:  # If fitnesses are equal, use a random tie breaker
                    rnum = random.uniform(0, 1)
                    if rnum > 0.5:
                        new_population["pol{0}".format(pol_id)] = copy.deepcopy(self.population["pol{0}".format(p1)])
                    else:
                        new_population["pol{0}".format(pol_id)] = copy.deepcopy(self.population["pol{0}".format(p2)])

        self.population = {}
        self.population = copy.deepcopy(new_population)

    def epsilon_greedy_select(self):  # Choose K solutions
        """
        Select parents using e-greedy selection
        """
        new_population = {}
        for pol_id in range(self.pop_size):
            if pol_id < self.n_elites:
                new_population["pol{0}".format(pol_id)] = copy.deepcopy(self.population["pol{0}".format(pol_id)])
            else:
                rnum = random.uniform(0, 1)
                if rnum < self.eps:  # Greedy Selection
                    max_index = np.argmax(self.fitness)
                    new_population["pol{0}".format(pol_id)] = copy.deepcopy(self.population["pol{0}".format(max_index)])
                else:  # Random Selection
                    parent = random.randint(1, (self.pop_size - 1))
                    new_population["pol{0}".format(pol_id)] = copy.deepcopy(self.population["pol{0}".format(parent)])

        self.population = {}
        self.population = copy.deepcopy(new_population)

    def random_selection(self):
        """
        Choose next generation of policies using elite-random selection
        """
        new_population = {}
        for pol_id in range(self.pop_size):
            if pol_id < self.n_elites:
                new_population["pol{0}".format(pol_id)] = copy.deepcopy(self.population["pol{0}".format(pol_id)])
            else:
                parent = random.randint(0, self.pop_size-1)
                new_population["pol{0}".format(pol_id)] = copy.deepcopy(self.population["pol{0}".format(parent)])

        self.population = {}
        self.population = copy.deepcopy(new_population)

    def rank_population(self):
        """
        Reorders the population in terms of fitness (high to low)
        """
        ranked_population = copy.deepcopy(self.population)
        for pol_a in range(self.pop_size):
            pol_b = pol_a + 1
            ranked_population["pol{0}".format(pol_a)] = copy.deepcopy(self.population["pol{0}".format(pol_a)])
            while pol_b < (self.pop_size):
                if pol_a != pol_b:
                    if self.fitness[pol_a] < self.fitness[pol_b]:
                        self.fitness[pol_a], self.fitness[pol_b] = self.fitness[pol_b], self.fitness[pol_a]
                        ranked_population["pol{0}".format(pol_a)] = copy.deepcopy(self.population["pol{0}".format(pol_b)])
                pol_b += 1

        self.population = {}
        self.population = copy.deepcopy(ranked_population)

    def down_select(self):  # Create a new offspring population using parents from top 50% of policies
        """
        Select parents create offspring population, and perform mutation operations
        """
        self.rank_population()
        self.epsilon_greedy_select()  # Select K successors using epsilon greedy
        # self.binary_tournament_selection()
        # self.random_selection()  # Select k successors using fit prop selection
        self.weight_mutate()  # Mutate successors

