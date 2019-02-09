import math
cimport cython
from parameters import Parameters as p
import numpy as np


cpdef one_of_each_type(c_number):
    cdef int n_rovers = p.num_rovers
    cdef int n_types = p.num_types
    cdef double[:, :] partners = np.zeros((c_number, 2))
    cdef double act_dist = p.activation_dist
    cdef int i

    for i in range(c_number):
        partners[i, 0] = 3.0
        partners[i, 1] = float(i)  # Type of rover

    return partners
