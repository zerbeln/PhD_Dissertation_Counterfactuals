from parameters import parameters as p
from standard_rover_domain import rover_global as global_nav
from standard_rover_domain import rover_difference as difference_nav
from standard_rover_domain import rover_dpp as dpp_nav
from standard_rover_cba import rover_global as global_skills
from standard_rover_cba import rover_difference as difference_skills
from standard_rover_cba import rover_dpp as dpp_skills
from ACG.acg import train_supervisor


if __name__ == '__main__':
    """
    Run code to train rovers and supervisors with ACG
    """

    assert (p["algorithm"] == "ACG_Skills" or p["algorithm"] == "ACG_Nav")

    if p["algorrithm"] == "ACG_Nav":
        if p["acg_alg"] == "Global":
            print("Training Rovers: Global Rewards")
            global_nav()
        elif p["acg_alg"] == "Difference":
            print("Training Rovers: Difference Rewards")
            difference_nav()
        elif p["acg_alg"] == "DPP":
            print("Training Rovers: D++ Rewards")
            dpp_nav()
        else:
            print("ALGORITHM TYPE ERROR")
    else:
        if p["acg_alg"] == "Global":
            print("Training Rovers: Global Rewards")
            global_skills()
        elif p["acg_alg"] == "Difference":
            print("Training Rovers: Difference Rewards")
            difference_skills()
        elif p["acg_alg"] == "DPP":
            print("Training Rovers: D++ Rewards")
            dpp_skills()
        else:
            print("ALGORITHM TYPE ERROR")

    print("Training Supervisor")
    train_supervisor()
