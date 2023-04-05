from parameters import parameters as p


def generate_high_low_counterfactuals(pois):
    counterfactuals = []
    for i in range(p["n_poi"]):
        counterfactuals.append([])
        for j in range(p["n_rovers"]):
            if pois[f'P{i}'].value <= 5 and j % 2 == 0:
                counterfactuals[i].append(1)
            elif pois[f'P{i}'].value > 5 and j % 2 != 0:
                counterfactuals[i].append(1)
            else:
                counterfactuals[i].append(0)

    return counterfactuals


def ten_poi_counterfactuals():
    # Define Expert Counterfactuals
    cp0 = [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0]  # 10
    cp1 = [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0]  # 3
    cp2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 8
    cp3 = [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1]  # 9
    cp4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 5
    cp5 = [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1]  # 5
    cp6 = [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1]  # 7
    cp7 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 8
    cp8 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 5
    cp9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 5

    counterfactuals = [cp0, cp1, cp2, cp3, cp4, cp5, cp6, cp7, cp8, cp9]

    return counterfactuals


def four_poi_counterfactuals():
    # Define Expert Counterfactuals
    cp1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 10
    cp2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 8
    cp3 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 5
    cp4 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 4

    counterfactuals = [cp1, cp2, cp3, cp4]

    return counterfactuals


def five_poi_counterfactuals():
    # Define Expert Counterfactuals
    cp1 = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]  # 9
    cp2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 5
    cp3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 7
    cp4 = [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]  # 4
    cp5 = [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]  # 10

    counterfactuals = [cp1, cp2, cp3, cp4, cp5]

    return counterfactuals


def two_poi_counterfactuals():
    # Define Expert Counterfactuals
    cp1 = [1, 1, 1, 1, 1, 1]  # 10
    cp2 = [0, 0, 0, 0, 0, 0]  # 4

    counterfactuals = [cp1, cp2]

    return counterfactuals

