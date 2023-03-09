from parameters import parameters as p
from standard_rover_domain import rover_global, rover_difference, rover_dpp
from CFL.cfl import rover_cdpp, rover_cdif


if __name__ == '__main__':
    """
    Run classic or tightly coupled rover domain using either G, D, D++, or CFL
    This file trains rover policies directly (not skills based)
    """

    assert (p["algorithm"] != "CKI" and p["algorithm"] != "ACG")

    if p["algorithm"] == "Global":
        print("Rover Domain: Global Rewards")
        rover_global()
    elif p["algorithm"] == "Difference":
        print("Rover Domain: Difference Rewards")
        rover_difference()
    elif p["algorithm"] == "DPP":
        print("Rover Domain: D++ Rewards")
        rover_dpp()
    elif p["algorithm"] == "CFL":
        print("Rover Domain: CFL")
        rover_cdpp()
    else:
        print("ALGORITHM TYPE ERROR")
