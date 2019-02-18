from parameters import Parameters as p
import numpy as np
import random


def one_of_each_type(c_number, current_type):
    partners = np.zeros((c_number, 2))

    for i in range(c_number):
        partners[i, 0] = p.activation_dist - 1
        partners[i, 1] = float(i)  # Type of rover

        if p.num_types > 1:
            while partners[i, 1] == current_type:  # Do not suggest same type if there are multiple
                partners[i, 1] = random.randint(0, (p.num_types-1))

    return partners
