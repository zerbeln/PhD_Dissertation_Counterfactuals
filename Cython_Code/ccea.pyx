import numpy as np
from AADI_RoverDomain.parameters import Parameters as p
import random


cdef class Ccea:

    # Declare variables
    cdef int parent_psize
    cdef int offspring_psize
    cdef int total_pop_size
    cdef int policy_size
    cdef int n_populations
    cdef double percent_mut
    cdef double mut_rate
    cdef double eps
    cdef double[:, :, :] parent_pop
    cdef double[:, :, :] offspring_pop
    cdef public double[:, :, :] pops
    cdef public double[:, :] fitness
    cdef public double[:, :] team_selection

    def __cinit__(self):
        self.parent_psize = p.parent_pop_size
        self.offspring_psize = p.offspring_pop_size
        self.total_pop_size = p.parent_pop_size + p.offspring_pop_size
        self.policy_size = (p.num_inputs + 1)*p.num_nodes + (p.num_nodes + 1) * p.num_outputs  # Number of weights for NN
        self.n_populations = p.num_rovers
        self.percent_mut = p.percentage_mut
        self.mut_rate = p.mutation_rate
        self.eps = p.epsilon
        self.pops = np.zeros((self.n_populations, self.total_pop_size, self.policy_size))
        self.parent_pop = np.zeros((self.n_populations, self.parent_psize, self.policy_size))
        self.offspring_pop = np.zeros((self.n_populations, self.offspring_psize, self.policy_size))
        self.fitness = np.zeros((self.n_populations, self.total_pop_size))
        self.team_selection = np.ones((self.n_populations, self.total_pop_size)) * (-1)

    cpdef reset_populations(self):  # Re-initializes CCEA populations for new run
        """
        Create new populations (for beginning of stat run)
        :return: None
        """
        cdef int pop_index, policy_index, w
        cdef double weight

        self.pops = np.zeros((self.n_populations, self.total_pop_size, self.policy_size))
        self.parent_pop = np.zeros((self.n_populations, self.parent_psize, self.policy_size))
        self.offspring_pop = np.zeros((self.n_populations, self.offspring_psize, self.policy_size))
        self.fitness = np.zeros((self.n_populations, self.total_pop_size))
        self.team_selection = np.ones((self.n_populations, self.total_pop_size)) * (-1)

        for pop_index in range(self.n_populations):
            for policy_index in range(self.parent_psize):
                for w in range(self.policy_size):
                    weight = np.random.normal(0, 1)
                    self.parent_pop[pop_index, policy_index, w] = weight
            for policy_index in range(self.offspring_psize):
                for w in range(self.policy_size):
                    weight = np.random.normal(0, 1)
                    self.offspring_pop[pop_index, policy_index, w] = weight

        self.combine_pops()

    cpdef select_policy_teams(self):  # Create policy teams for testing
        """
        Choose teams of individuals from among populations to be tested
        :return: None
        """
        cdef int pop_id, policy_id, k, rpol
        self.team_selection = np.ones((self.n_populations, self.total_pop_size)) * (-1)

        for pop_id in range(self.n_populations):
            for policy_id in range(self.total_pop_size):
                rpol = random.randint(0, (self.total_pop_size - 1))  # Select a random policy from pop
                k = 0
                while k < policy_id:  # Check for duplicates
                    if rpol == self.team_selection[pop_id, k]:
                        rpol = random.randint(0, (self.total_pop_size - 1))
                        k = -1
                    k += 1
                self.team_selection[pop_id, policy_id] = rpol  # Assign policy to team

    cpdef mutate(self):  # Mutate policy based on probability
        """
        Mutate offspring populations
        :return: None
        """
        cdef int pop_index, policy_index, mutate_n, target
        cdef double rnum, weight

        for pop_index in range(self.n_populations):
            policy_index = 0
            mutate_n = int(self.percent_mut * self.policy_size)
            if mutate_n == 0:
                mutate_n = 1
            while policy_index < self.offspring_psize:
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_rate:
                    for w in range(mutate_n):
                        target = random.randint(0, (self.policy_size - 1))  # Select random weight to mutate
                        weight = np.random.normal(0, 1)
                        self.offspring_pop[pop_index, policy_index, target] = weight
                policy_index += 1

    cpdef epsilon_greedy_select(self):  # Choose K solutions
        """
        Select parents from which an offspring population will be created
        :return: None
        """
        cdef int pop_id, policy_id, parent, pol_index
        cdef double rnum

        for pop_id in range(self.n_populations):
            policy_id = 0
            while policy_id < self.parent_psize:
                rnum = random.uniform(0, 1)
                if rnum >= self.eps:  # Choose best policy
                    pol_index = np.argmax(self.fitness[pop_id])
                    self.parent_pop[pop_id, policy_id] = self.pops[pop_id, pol_index].copy()
                else:
                    parent = random.randint(0, (self.total_pop_size-1))  # Choose a random parent
                    self.parent_pop[pop_id, policy_id] = self.pops[pop_id, parent].copy()
                policy_id += 1

    cpdef down_select(self):  # Create a new offspring population using parents from top 50% of policies
        """
        Select parent,s create offspring population, and perform mutation operations
        :return: None
        """
        self.epsilon_greedy_select()  # Select K solutions using epsilon greedy
        self.offspring_pop = self.parent_pop.copy()  # Produce K offspring
        self.mutate()  # Mutate offspring population
        self.combine_pops()

    cpdef combine_pops(self):
        """
        Combine parent and offspring populations into single population array
        :return: None
        """
        cdef pop_id, off_pol_id, pol_id

        for pop_id in range(self.n_populations):
            off_pol_id = 0
            for pol_id in range(self.total_pop_size):
                if pol_id < self.parent_psize:
                    self.pops[pop_id, pol_id] = self.parent_pop[pop_id, pol_id].copy()
                else:
                    self.pops[pop_id, pol_id] = self.offspring_pop[pop_id, off_pol_id].copy()
                    off_pol_id += 1
