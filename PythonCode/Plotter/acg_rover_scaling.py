import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from plots_common_functions import get_acg_standard_err


def generate_incursion_plots(sruns, n_rovers):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    # File Paths
    acg_fpath1 = '../1Rover/Output_Data/ACGHazardIncursions.csv'
    acg_fpath2 = '../2Rovers/Output_Data/ACGHazardIncursions.csv'
    acg_fpath3 = '../3Rovers/Output_Data/ACGHazardIncursions.csv'
    acg_fpath4 = '../4Rovers/Output_Data/ACGHazardIncursions.csv'
    acg_fpath5 = '../5Rovers/Output_Data/ACGHazardIncursions.csv'
    fpath1 = '../1Rover/Output_Data/HazardIncursions.csv'
    fpath2 = '../2Rovers/Output_Data/HazardIncursions.csv'
    fpath3 = '../3Rovers/Output_Data/HazardIncursions.csv'
    fpath4 = '../4Rovers/Output_Data/HazardIncursions.csv'
    fpath5 = '../5Rovers/Output_Data/HazardIncursions.csv'

    rover_incursions = np.zeros(n_rovers)
    rover_inc_err = []
    acg_incursions = np.zeros(n_rovers)
    acg_inc_err = []

    # Read data from CSV files
    config_input = []
    with open(acg_fpath1) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    with open(acg_fpath2) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    with open(acg_fpath3) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    with open(acg_fpath4) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    # with open(acg_fpath5) as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',')
    #     for row in csv_reader:
    #         config_input.append(row)

    test_id = 0  # Test ID
    average_incursions = np.zeros(int(sruns))  # Track number of incursions for each stat run of a given test
    for row in config_input:
        for i in range(int(sruns)):
            average_incursions[i] = float(row[i])

        acg_incursions[test_id] += np.mean(average_incursions)
        acg_inc_err.append(get_acg_standard_err(row, np.mean(average_incursions), sruns))
        test_id += 1

    # Read data from CSV files
    config_input = []
    with open(fpath1) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    with open(fpath2) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    with open(fpath3) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    with open(fpath4) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    # with open(fpath5) as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',')
    #     for row in csv_reader:
    #         config_input.append(row)

    test_id = 0  # Test ID
    average_incursions = np.zeros(int(sruns))  # Track number of incursions for each stat run of a given test
    for row in config_input:
        for i in range(int(sruns)):
            average_incursions[i] = float(row[i])

        rover_incursions[test_id] += np.mean(average_incursions)
        rover_inc_err.append(get_acg_standard_err(row, np.mean(average_incursions), sruns))
        test_id += 1

    # Plot The Data
    x_axis = np.arange(n_rovers)
    labels = [1, 2, 3, 4]
    width = 0.35
    fig, ax = plt.subplots()
    p1 = plt.barh(x_axis - width/2, rover_incursions,  width, xerr=rover_inc_err, color=color1, label="Without Supervisor")
    p2 = plt.barh(x_axis + width/2, acg_incursions, width, xerr=acg_inc_err, color=color4, label="With Supervisor")

    ax.set_ylabel("Agent Team Size")
    ax.set_xlabel("Average Number of Hazard Entries")
    ax.set_yticks(x_axis, labels)
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/ACG_Incursions.pdf")
    plt.close()


def generate_performance_graphs(sruns, n_rovers, scaling):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    # File Paths
    acg_fpath1 = '../1Rover/Output_Data/TeamPerformance_ACG.csv'
    acg_fpath2 = '../2Rovers/Output_Data/TeamPerformance_ACG.csv'
    acg_fpath3 = '../3Rovers/Output_Data/TeamPerformance_ACG.csv'
    acg_fpath4 = '../4Rovers/Output_Data/TeamPerformance_ACG.csv'
    acg_fpath5 = '../5Rovers/Output_Data/TeamPerformance_ACG.csv'
    global_fpath1 = '../1Rover/Output_Data/Final_GlobalRewards.csv'
    global_fpath2 = '../2Rovers/Output_Data/Final_GlobalRewards.csv'
    global_fpath3 = '../3Rovers/Output_Data/Final_GlobalRewards.csv'
    global_fpath4 = '../4Rovers/Output_Data/Final_GlobalRewards.csv'
    global_fpath5 = '../5Rovers/Output_Data/Final_GlobalRewards.csv'

    rover_performance = np.zeros(n_rovers)
    rov_err = []
    acg_performance = np.zeros(n_rovers)
    acg_err = []

    config_input = []
    with open(acg_fpath1) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    with open(acg_fpath2) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    with open(acg_fpath3) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    with open(acg_fpath4) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    # with open(acg_fpath5) as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',')
    #     for row in csv_reader:
    #         config_input.append(row)

    test_id = 0
    team_rewards = np.zeros(int(sruns))
    for row in config_input:
        for i in range(sruns):
            team_rewards[i] = float(row[i])
        acg_performance[test_id] = np.mean(team_rewards) + scaling
        acg_err.append(get_acg_standard_err(row, np.mean(team_rewards), sruns))
        test_id += 1

    config_input = []
    with open(global_fpath1) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    with open(global_fpath2) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    with open(global_fpath3) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    with open(global_fpath4) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            config_input.append(row)
    # with open(global_fpath5) as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',')
    #     for row in csv_reader:
    #         config_input.append(row)

    test_id = 0
    team_rewards = np.zeros(int(sruns))
    for row in config_input:
        for i in range(sruns):
            team_rewards[i] = float(row[i])
        rover_performance[test_id] = np.mean(team_rewards) + scaling
        rov_err.append(get_acg_standard_err(row, np.mean(team_rewards), sruns))
        test_id += 1

    # Plot The Data
    x_line = 0
    x_axis = np.arange(n_rovers)
    labels = [1, 2, 3, 4]
    width = 0.35
    fig, ax = plt.subplots()
    p1 = plt.bar(x_axis - width / 2, rover_performance, width, yerr=rov_err, color=color1, label="Without Supervisor")
    p2 = plt.bar(x_axis + width / 2, acg_performance, width, yerr=acg_err, color=color4, label="With Supervisor")

    ax.set_xlabel("Agent Team Size")
    ax.set_ylabel("Average Team Performance")
    ax.set_xticks(x_axis, labels)
    ax.legend(ncol=2, loc=1, bbox_to_anchor=(1.00, 1.09))
    fig.tight_layout()
    plt.axhline(x_line, color='black')

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
    n_rovers = 4
    scaling = 0.0

    generate_performance_graphs(sruns, n_rovers, scaling)
    generate_incursion_plots(sruns, n_rovers)
