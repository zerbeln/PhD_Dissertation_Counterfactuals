from parameters import parameters as p
# from run_standard_skills import rover_global, rover_difference, rover_dpp
from standard_rover_domain import rover_global, rover_difference, rover_dpp
from CBA.cba import train_cba
from CFL.cfl import rover_sdpp


if __name__ == '__main__':
    """
    Run classic rove domain using either G, D, or D++ for reward feedback.
    """

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
    elif p["algorithm"] == "CFL":
        suggestions = [0, 0, 0]
        rover_sdpp(suggestions)
    else:
        print("ALGORITHM TYPE ERROR")
