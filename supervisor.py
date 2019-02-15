from parameters import Parameters as p
import numpy as np


def one_of_each_type(c_number):
    partners = np.zeros((c_number, 2))

    for i in range(c_number):
        partners[i, 0] = 3.00
        partners[i, 1] = float(i)  # Type of rover

    return partners
