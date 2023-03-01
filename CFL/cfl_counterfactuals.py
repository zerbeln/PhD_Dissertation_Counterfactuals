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

    assert (len(counterfactuals) == p["n_poi"])
    for i in range(p["n_poi"]):
        assert (len(counterfactuals[i]) == p["n_rovers"])

    return counterfactuals


def generate_custom_counterfactuals():
    # Define Expert Counterfactuals
    cp1 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 10
    cp2 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 3
    cp3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 8
    cp4 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 9
    cp5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 5
    cp6 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 5
    cp7 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 7
    cp8 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 8
    cp9 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 5
    cp10 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 5

    counterfactuals = [cp1, cp2, cp3, cp4, cp5, cp6, cp7, cp8, cp9, cp10]
    assert(len(counterfactuals) == p["n_poi"])
    for i in range(p["n_poi"]):
        assert (len(counterfactuals[i]) == p["n_rovers"])

    return counterfactuals


def generate_two_poi_counterfactuals():
    assert(p["n_poi"] == 2)
    # Define Expert Counterfactuals
    cp1 = [1, 1, 1, 1, 1, 1]  # 10
    cp2 = [0, 0, 0, 0, 0, 0]  # 4

    counterfactuals = [cp1, cp2]

    return counterfactuals

