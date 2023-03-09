from parameters import parameters as p
from standard_rover_cki import rover_global, rover_difference, rover_dpp
from CKI.cki import train_cki


if __name__ == '__main__':
    """
    Run classic or tightly coupled rover domain using either G, D, D++, or CKI with pre-defined rover skills.
    """

    # This file trains rovers to use skill selection instead of training the skills themselves
    assert(p["algorithm"] != "CFL")
    assert (p["algorithm"] != "ACG")

    if p["algorithm"] == "Global":
        print("Rover Domain: Global Rewards")
        rover_global()
    elif p["algorithm"] == "Difference":
        print("Rover Domain: Difference Rewards")
        rover_difference()
    elif p["algorithm"] == "DPP":
        print("Rover Domain: D++ Rewards")
        rover_dpp()
    elif p["algorithm"] == "CKI":
        print("Rover Domain: CKI with custom skills")
        train_cki()
    else:
        print("ALGORITHM TYPE ERROR")
