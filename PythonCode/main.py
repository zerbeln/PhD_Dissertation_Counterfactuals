from parameters import parameters as p
from standard_rover_domain import rover_global, rover_difference, rover_dpp
from CFL.cfl import rover_sdpp


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
        print("Rover Domain: CFL")
        if p["counterfactual_type"] == "High":
            counterfactuals = [0 for i in range(p["n_rovers"])]
        elif p["counterfactual_type"] == "Low":
            counterfactuals = [1 for i in range(p["n_rovers"])]
        else:
            print("COUNTERFACTUAL TYPE ERROR")
        rover_sdpp(counterfactuals)
    else:
        print("ALGORITHM TYPE ERROR")
