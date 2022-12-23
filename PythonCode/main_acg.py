from parameters import parameters as p
from standard_rover_cba import rover_global as global_skills
from standard_rover_cba import rover_difference as difference_skills
from standard_rover_cba import rover_dpp as dpp_skills
from ACG.acg import train_supervisor_poi_hazards, train_supervisor_rover_loss


if __name__ == '__main__':
    """
    Run code to train rovers and supervisors with ACG
    """

    assert (p["algorithm"] == "ACG")

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

    if p["sup_train"] == "Hazards":
        print("Training Supervisor: Hazards")
        train_supervisor_poi_hazards()
    elif p["sup_train"] == "Rover_Loss":
        print("Training Supervisor: Rover Loss")
        train_supervisor_rover_loss()
    else:
        print("INCORRECT SUPERVISOR TRAINING TYPE")
