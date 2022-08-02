from parameters import parameters as p
from run_standard import rover_global, rover_difference, rover_dpp
from CBA.cba import train_cba


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
    else:
        print("ALGORITHM TYPE ERROR")
