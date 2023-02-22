from parameters import parameters as p
from standard_rover_cba import rover_global, rover_difference, rover_dpp
from CKI.cki import train_cki


if __name__ == '__main__':
    """
    Run classic or tightly coupled rover domain using either G, D, D++, or CKI with pre-defined rover skills.
    """

    assert(p["algorithm"] != "CFL")  # This main file is for use with pre-defined rover skills only
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
