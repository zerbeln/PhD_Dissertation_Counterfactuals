import numpy as np
import random


cdef class Ccea:
    # Declare variables
    cdef int parent_psize
    cdef int offspring_psize
    cdef public int total_pop_size
    cdef int policy_size
    cdef double mut_rate
    cdef double mut_chance
    cdef double eps
    cdef double[:, :] parent_pop
    cdef double[:, :] offspring_pop
    cdef public double[:, :] pops
    cdef public double[:] fitness
    cdef public double[:] team_selection

    def __cinit__(self, object p):
        self.parent_psize = int(p.parent_pop_size)
        self.offspring_psize = int(p.offspring_pop_size)
        self.total_pop_size = int(p.parent_pop_size + p.offspring_pop_size)
        self.policy_size = int((p.num_inputs + 1)*p.num_nodes + (p.num_nodes + 1) * p.num_outputs)  # Number of weights for NN
        self.mut_rate = p.mutation_rate
        self.mut_chance = p.mutation_prob
        self.eps = p.epsilon
        self.pops = np.zeros((self.total_pop_size, self.policy_size))
        self.parent_pop = np.zeros((self.parent_psize, self.policy_size))
        self.offspring_pop = np.zeros((self.offspring_psize, self.policy_size))
        self.fitness = np.zeros(self.total_pop_size)
        self.team_selection = np.ones(self.total_pop_size) * (-1)

    cpdef reset_population(self):  # Re-initializes CCEA populations for new run
        """
        Create new populations (for beginning of stat run)
        :return: None
        """
        cdef int policy_index, w
        cdef double weight

        self.pops = np.zeros((self.total_pop_size, self.policy_size))
        self.parent_pop = np.zeros((self.parent_psize, self.policy_size))
        self.offspring_pop = np.zeros((self.offspring_psize, self.policy_size))
        self.fitness = np.zeros(self.total_pop_size)
        self.team_selection = np.ones(self.total_pop_size) * (-1)

        for policy_index in range(self.parent_psize):
            for w in range(self.policy_size):
                weight = np.random.normal(0, 1)
                self.parent_pop[policy_index, w] = weight
        for policy_index in range(self.offspring_psize):
            for w in range(self.policy_size):
                weight = np.random.normal(0, 1)
                self.offspring_pop[policy_index, w] = weight

        self.combine_pops()

    cpdef select_policy_teams(self):  # Create policy teams for testing
        """
        Choose teams of individuals from among populations to be tested
        :return: None
        """
        cdef int policy_id, k, target
        self.team_selection = np.ones(self.total_pop_size) * (-1)

        for policy_id in range(self.total_pop_size):
            target = random.randint(0, (self.total_pop_size - 1))  # Select a random policy from pop
            k = 0
            while k < policy_id:  # Check for duplicates
                if target == self.team_selection[k]:
                    target = random.randint(0, (self.total_pop_size - 1))
                    k = -1
                k += 1
            self.team_selection[policy_id] = target  # Assign policy to team

    cpdef weight_mutate(self):
        """
        Mutate offspring populations (each weight has a probability of mutation)
        :return: 
        """
        cdef int pol_id, w
        cdef double rnum, mutation

        for pol_id in range(self.offspring_psize):
            for w in range(self.policy_size):
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_chance:
                    mutation = (np.random.normal(0, self.mut_rate) * self.offspring_pop[pol_id, w])
                    self.offspring_pop[pol_id, w] += mutation

    cpdef epsilon_greedy_select(self):  # Choose K solutions
        """
        Select parents from which an offspring population will be created
        :return: None
        """
        cdef int policy_id, parent, pol_index
        cdef double rnum

        policy_id = 0
        while policy_id < self.parent_psize:
            rnum = random.uniform(0, 1)
            if rnum > self.eps:  # Choose best policy
                pol_index = np.argmax(self.fitness)
                self.parent_pop[policy_id] = self.pops[pol_index].copy()
            else:
                parent = random.randint(1, (self.total_pop_size-1))  # Choose a random parent
                self.parent_pop[policy_id] = self.pops[parent].copy()
            policy_id += 1

    cpdef down_select(self):  # Create a new offspring population using parents from top 50% of policies
        """
        Select parent,s create offspring population, and perform mutation operations
        :return: None
        """
        self.rank_individuals()
        self.epsilon_greedy_select()  # Select K successors using epsilon greedy
        self.offspring_pop = self.parent_pop.copy()
        self.weight_mutate()  # Mutate successors
        self.combine_pops()

    cpdef rank_individuals(self):
        """
        Order individuals in the population based on their fitness scores
        :return: None
        """
        cdef int pol_id_a, pol_id_b

        for pol_id_a in range(self.total_pop_size-1):
            pol_id_b = pol_id_a + 1
            while pol_id_b < (self.total_pop_size):
                if pol_id_a != pol_id_b:
                    if self.fitness[pol_id_a] < self.fitness[pol_id_b]:
                        self.fitness[pol_id_a], self.fitness[pol_id_b] = self.fitness[pol_id_b], self.fitness[pol_id_a]
                        self.pops[pol_id_a], self.pops[pol_id_b] = self.pops[pol_id_b], self.pops[pol_id_a]
                pol_id_b += 1


    cpdef combine_pops(self):
        """
        Combine parent and offspring populations into single population array
        :return: None
        """
        cdef int off_pol_id, pol_id

        off_pol_id = 0
        for pol_id in range(self.total_pop_size):
            if pol_id < self.parent_psize:
                self.pops[pol_id] = self.parent_pop[pol_id].copy()
            else:
                self.pops[pol_id] = self.offspring_pop[off_pol_id].copy()
                off_pol_id += 1

    cpdef reset_fitness(self):
        self.fitness = np.zeros(self.total_pop_size)
