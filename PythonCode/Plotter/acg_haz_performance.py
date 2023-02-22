import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from plots_common_functions import get_acg_standard_err


def generate_performance_graphs(sruns, n_tests, scaling):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    # File Paths
    acg_fpath1 = '../0Hazards/3Rovers/Output_Data/TeamPerformance_ACG.csv'
    acg_fpath2 = '../1Hazard/3Rovers/Output_Data/TeamPerformance_ACG.csv'
    acg_fpath3 = '../2Hazards/3Rovers/Output_Data/TeamPerformance_ACG.csv'
    acg_fpath4 = '../3Hazards/3Rovers/Output_Data/TeamPerformance_ACG.csv'
    acg_fpath5 = '../4Hazards/3Rovers/Output_Data/TeamPerformance_ACG.csv'
    global_fpath1 = '../0Hazards/3Rovers/Output_Data/Final_GlobalRewards.csv'
    global_fpath2 = '../1Hazard/3Rovers/Output_Data/Final_GlobalRewards.csv'
    global_fpath3 = '../2Hazards/3Rovers/Output_Data/Final_GlobalRewards.csv'
    global_fpath4 = '../3Hazards/3Rovers/Output_Data/Final_GlobalRewards.csv'
    global_fpath5 = '../4Hazards/3Rovers/Output_Data/Final_GlobalRewards.csv'

    rover_performance = np.zeros(n_tests)
    rov_err = []
    acg_performance = np.zeros(n_tests)
    acg_err = []

    acg_input = []
    with open(acg_fpath1) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            acg_input.append(row)
    with open(acg_fpath2) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            acg_input.append(row)
    with open(acg_fpath3) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            acg_input.append(row)
    with open(acg_fpath4) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            acg_input.append(row)
    with open(acg_fpath5) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            acg_input.append(row)

    test_id = 0
    team_rewards = np.zeros(int(sruns))
    for row in acg_input:
        for i in range(sruns):
            team_rewards[i] = float(row[i])
        acg_performance[test_id] = np.mean(team_rewards) + scaling
        acg_err.append(get_acg_standard_err(row, np.mean(team_rewards), sruns))
        test_id += 1

    rov_input = []
    with open(global_fpath1) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input.append(row)
    with open(global_fpath2) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input.append(row)
    with open(global_fpath3) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input.append(row)
    with open(global_fpath4) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input.append(row)
    with open(global_fpath5) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input.append(row)

    test_id = 0
    team_rewards = np.zeros(int(sruns))
    for row in rov_input:
        for i in range(sruns):
            team_rewards[i] = float(row[i])
        rover_performance[test_id] = np.mean(team_rewards) + scaling
        rov_err.append(get_acg_standard_err(row, np.mean(team_rewards), sruns))
        test_id += 1

    # Plot The Data
    x_axis = np.arange(n_tests)
    labels = [0, 1, 2, 3, 4]
    plt.errorbar(x_axis, rover_performance, rov_err, linestyle="-", marker='o', color=color1, label="Without Supervisor")
    plt.errorbar(x_axis, acg_performance, acg_err, linestyle="--", marker='s', color=color4, label="With Supervisor")

    plt.xlabel("Number of Hazards")
    plt.ylabel("Average Team Performance")
    plt.xticks(x_axis, labels)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/ACG_Performance.pdf")
    plt.close()


if __name__ == '__main__':
    """
    This code generates the performance plots for the counterfactual supervisor.
    The functions plot overall team performance according to G and the number of entries into hazards.
    Requires the number of stat runs and the reward type used to train agents as inputs when calling script.
    """

    sruns = 30
    sample_rate = 20
    n_tests = 5
    scaling = 0.0

    generate_performance_graphs(sruns, n_tests, scaling)
