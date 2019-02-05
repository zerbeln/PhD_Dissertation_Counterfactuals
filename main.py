# distutils: language = c++
# cython: language_level=3, boundscheck=True

import pyximport; pyximport.install()
from ccea import Ccea
from neural_network import NeuralNetwork
from parameters import Parameters as p
from rover_domain_w_setup import *
from rover_domain import RoverDomain


def main():
    cc = Ccea()
    nn = NeuralNetwork()
    rd = RoverDomain()

    # Set Test Parameters
    rd.n_rovers = p.num_rovers
    rd.n_pois = p.num_pois
    rd.n_steps = p.num_steps

main()  # Run the program