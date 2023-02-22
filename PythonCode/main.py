from parameters import parameters as p
from standard_rover_domain import rover_global, rover_difference, rover_dpp
from CFL.cfl import rover_cdpp, rover_cdif


if __name__ == '__main__':
    """
    Run classic or tightly coupled rover domain using either G, D, D++, or CFL
    """

    assert (p["algorithm"] != "CKI")  # This main file is for use with rovers learning navigation (not skills)

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

        # Define Expert Counterfactuals
        counterfacutals = []
        cp1 = [1, 1, 1, 1, 1, 1]
        cp2 = [1, 1, 1, 1, 1, 1]
        cp3 = [0, 0, 0, 0, 0, 0]
        cp4 = [0, 0, 0, 0, 0, 0]
        counterfactuals = [cp1, cp2, cp3, cp4]

        # Check that this parameter was manually tuned for the specific experiment
        assert(len(counterfactuals) == p["n_poi"])
        for cfact in counterfactuals:
            assert(len(cfact) == p["n_rovers"])

        rover_cdpp(counterfactuals)
    else:
        print("ALGORITHM TYPE ERROR")
