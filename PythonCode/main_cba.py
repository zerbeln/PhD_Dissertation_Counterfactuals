from parameters import parameters as p
from standard_rover_skills import rover_global, rover_difference, rover_dpp
from CBA.cba import train_cba


if __name__ == '__main__':
    """
    Run classic or tightly coupled rover domain using either G, D, D++, or CBA with pre-defined rover skills.
    """

    assert(p["algorithm"] != "CFL")  # This main file is for use with pre-defined rover skills only

    if p["algorithm"] == "Global":
        print("Rover Domain: Global Rewards")
        rover_global()
    elif p["algorithm"] == "Difference":
        print("Rover Domain: Difference Rewards")
        rover_difference()
    elif p["algorithm"] == "DPP":
        print("Rover Domain: D++ Rewards")
        rover_dpp()
    elif p["algorithm"] == "CBA":
        print("Rover Domain: CBA with custom skills")
        train_cba()
    else:
        print("ALGORITHM TYPE ERROR")
