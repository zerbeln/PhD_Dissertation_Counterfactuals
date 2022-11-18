from parameters import parameters as p
from standard_rover_domain import rover_global, rover_difference, rover_dpp
from ACG.acg import train_supervisor


if __name__ == '__main__':
    """
    Run code to train rovers and supervisors with ACG
    """

    assert (p["algorithm"] == "ACG")  # This main file is for use with rovers learning navigation (not skills)

    if p["acg_alg"] == "Global":
        print("Training Rovers: Global Rewards")
        rover_global()
    elif p["acg_alg"] == "Difference":
        print("Training Rovers: Difference Rewards")
        rover_difference()
    elif p["acg_alg"] == "DPP":
        print("Training Rovers: D++ Rewards")
        rover_dpp()
    else:
        print("ALGORITHM TYPE ERROR")

    print("Training Supervisor")
    train_supervisor()
