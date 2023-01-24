import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import sys


def generate_incursion_plots(sruns, reward_type, n_tests):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    # File Paths
    fpath1 = '../OneConfig/Output_Data/HazardIncursions.csv'
    fpath2 = '../TwoConfigs/Output_Data/HazardIncursions.csv'
    fpath3 = '../FourConfigs/Output_Data/HazardIncursions.csv'
    fpath4 = '../C4/Output_Data/HazardIncursions.csv'
    fpath5 = '../5Rovers/Output_Data/HazardIncursions.csv'

    rover_incursions = np.zeros(n_tests)
    acg_incursions = np.zeros(n_tests)

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
    # with open(fpath4) as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',')
    #     for row in csv_reader:
    #         config_input.append(row)
    # with open(fpath5) as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',')
    #     for row in csv_reader:
    #         config_input.append(row)

    it_cnt = 0  # Iteration counter
    test_id = 0  # Test ID
    average_incursions = np.zeros(int(sruns))  # Track number of incursions for each stat run of a given test
    for row in config_input:
        for i in range(int(sruns)):
            average_incursions[i] = float(row[i])
            if average_incursions[i] == 0:
                average_incursions[i] = 0.1
        if it_cnt % 2 == 0:
            rover_incursions[test_id] += np.mean(average_incursions)
        else:
            acg_incursions[test_id] += np.mean(average_incursions)
            test_id += 1
        it_cnt += 1

    # Plot The Data
    x_axis = np.arange(n_tests)
    labels = [1, 2, 4]
    width = 0.35
    fig, ax = plt.subplots()
    p1 = plt.bar(x_axis - width/2, rover_incursions, width, color=color1, label=reward_type)
    p2 = plt.bar(x_axis + width/2, acg_incursions, width, color=color4, label="ACG")

    # ax.set_xlabel("Number of Rovers")
    ax.set_xlabel("Number of Training Configurations")
    ax.set_ylabel("Number of Rover Incursions")
    ax.set_xticks(x_axis, labels)
    ax.legend()
    fig.tight_layout()

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/ACG_Incursions.pdf")
    plt.close()


def generate_performance_graphs(sruns, reward_type, n_tests):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    # File Paths
    acg_fpath1 = '../OneConfig/Output_Data/TeamPerformance_ACG.csv'
    acg_fpath2 = '../TwoConfigs/Output_Data/TeamPerformance_ACG.csv'
    acg_fpath3 = '../FourConfigs/Output_Data/TeamPerformance_ACG.csv'
    acg_fpath4 = '../C4/Output_Data/TeamPerformance_ACG.csv'
    acg_fpath5 = '../5Rovers/Output_Data/TeamPerformance_ACG.csv'
    global_fpath1 = '../OneConfig/Output_Data/Final_GlobalRewards.csv'
    global_fpath2 = '../TwoConfigs/Output_Data/Final_GlobalRewards.csv'
    global_fpath3 = '../FourConfigs/Output_Data/Final_GlobalRewards.csv'
    global_fpath4 = '../C4/Output_Data/Final_GlobalRewards.csv'
    global_fpath5 = '../5Rovers/Output_Data/Final_GlobalRewards.csv'

    rover_performance = np.zeros(n_tests)
    acg_performance = np.zeros(n_tests)

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
    # with open(acg_fpath4) as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',')
    #     for row in csv_reader:
    #         config_input.append(row)
    # with open(acg_fpath5) as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',')
    #     for row in csv_reader:
    #         config_input.append(row)

    test_id = 0
    team_rewards = np.zeros(int(sruns))
    for row in config_input:
        for i in range(sruns):
            team_rewards[i] = float(row[i])
        acg_performance[test_id] = np.mean(team_rewards)
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
    # with open(global_fpath4) as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',')
    #     for row in csv_reader:
    #         config_input.append(row)
    # with open(global_fpath5) as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',')
    #     for row in csv_reader:
    #         config_input.append(row)

    test_id = 0
    team_rewards = np.zeros(int(sruns))
    for row in config_input:
        for i in range(sruns):
            team_rewards[i] = float(row[i])
        rover_performance[test_id] = np.mean(team_rewards)
        test_id += 1

    # Plot The Data
    x_axis = np.arange(n_tests)
    labels = [1, 2, 4]
    width = 0.35
    fig, ax = plt.subplots()
    p1 = plt.bar(x_axis - width / 2, rover_performance, width, color=color1, label=reward_type)
    p2 = plt.bar(x_axis + width / 2, acg_performance, width, color=color4, label="ACG")

    # ax.set_xlabel("Number of Rovers")
    ax.set_xlabel("Number of Training Configurations")
    ax.set_ylabel("Average Team Performance")
    ax.set_xticks(x_axis, labels)
    ax.legend()
    fig.tight_layout()

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/ACG_Performance.pdf")
    plt.close()


if __name__ == '__main__':
    sruns = int(sys.argv[1])
    reward_type = sys.argv[2]
    sample_rate = 20
    n_tests = 3

    generate_performance_graphs(sruns, reward_type, n_tests)
    generate_incursion_plots(sruns, reward_type, n_tests)
