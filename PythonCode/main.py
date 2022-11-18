from parameters import parameters as p
from standard_rover_domain import rover_global, rover_difference, rover_dpp
from CFL.cfl import rover_sdpp
from ACG.acg import train_supervisor


if __name__ == '__main__':
    """
    Run classic or tightly coupled rover domain using either G, D, D++, or CFL
    """

    assert (p["algorithm"] != "CBA")  # This main file is for use with rovers learning navigation (not skills)

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
        counterfactuals = [0, 0, 0]
        rover_sdpp(counterfactuals)
    elif p["algorithm"] == "ACG":
        print("Training Rovers")
        rover_global()
        print("Training Supervisor")
        train_supervisor()
    else:
        print("ALGORITHM TYPE ERROR")
