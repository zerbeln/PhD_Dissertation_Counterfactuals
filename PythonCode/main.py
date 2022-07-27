from parameters import parameters as p
from CBA.cba import train_cba_custom_skills, train_cba_learned_skills
from run_standard import rover_global, rover_difference, rover_dpp


if __name__ == '__main__':
    """
    Train suggestions interpreter (must have already pre-trained agent playbook)
    """

    if p["algorithm"] == "CBA" and p["custom_skills"]:
        print("Rover Domain: CBA with Custom Skills")
        train_cba_custom_skills()
    elif p["algorithm"] == "CBA" and not p["custom_skills"]:
        print("Rover Domain: CBA with Trained Skills")
        train_cba_learned_skills()
    elif p["algorithm"] == "Global":
        print("Rover Domain: Global Rewards")
        rover_global()
    elif p["algorithm"] == "Difference":
        print("Rover Domain: Difference Rewards")
        rover_difference()
    elif p["algorithm"] == "DPP":
        print("Rover Domain: D++ Rewards")
        rover_dpp()
    else:
        print("ALGORITHM TYPE ERROR")
