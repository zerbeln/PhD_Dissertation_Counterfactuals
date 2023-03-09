import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import sys
import csv
from plots_common_functions import import_reward_data, get_standard_err_learning


def generate_acg_learning_curves(generations, sample_rate, sruns):

    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    # Graph Data
    acg_file_path = '../Output_Data/ACG_Rewards.csv'
    acg_reward = import_reward_data(acg_file_path, generations, sample_rate, sruns)
    acg_stdev = get_standard_err_learning(acg_file_path, acg_reward, generations, sample_rate, sruns)

    x_axis = []
    for i in range(generations):
        if i % sample_rate == 0 or i == generations-1:
            x_axis.append(i)
    x_axis = np.array(x_axis)

    # Plot of Data
    plt.plot(x_axis, acg_reward, color=color4)

    # Plot of Error
    alpha_val = 0.2
    plt.fill_between(x_axis, acg_reward + acg_stdev, acg_reward - acg_stdev, alpha=alpha_val, facecolor=color4)

    # Graph Details
    plt.xlabel("Generations")
    plt.ylabel("Global Reward")
    plt.legend(["ACG Rewards"])

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/ACG_Learning_Curve.pdf")
    plt.close()


def generate_policy_learning_curves(generations, sample_rate, sruns, reward_type):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia
    alpha_val = 0.2

    x_axis = []
    for i in range(generations):
        if i % sample_rate == 0 or i == generations - 1:
            x_axis.append(i)
    x_axis = np.array(x_axis)

    # Graph Data
    if reward_type == "Global":
        g_file_path ='../Output_Data/Global_Reward.csv'
        g_reward = import_reward_data(g_file_path, generations, sample_rate, sruns)
        g_stdev = get_standard_err_learning(g_file_path, g_reward, generations, sample_rate, sruns)

        # Plot of Data
        plt.plot(x_axis, g_reward, color=color1)
        # Plot of Error
        plt.fill_between(x_axis, g_reward + g_stdev, g_reward - g_stdev, alpha=alpha_val, facecolor=color1)

    elif reward_type == "Difference":
        d_file_path = '../Output_Data/Difference_Reward.csv'
        d_reward = import_reward_data(d_file_path, generations, sample_rate, sruns)
        d_stdev = get_standard_err_learning(d_file_path, d_reward, generations, sample_rate, sruns)

        # Plot of Data
        plt.plot(x_axis, d_reward, color=color2)
        # Plot of Error
        plt.fill_between(x_axis, d_reward + d_stdev, d_reward - d_stdev, alpha=alpha_val, facecolor=color2)

    elif reward_type == "DPP":
        dpp_file_path = '../Output_Data/DPP_Reward.csv'
        dpp_reward = import_reward_data(dpp_file_path, generations, sample_rate, sruns)
        dpp_stdev = get_standard_err_learning(dpp_file_path, dpp_reward, generations, sample_rate, sruns)

        # Plot of Data
        plt.plot(x_axis, dpp_reward, color=color3)
        # Plot of Error
        plt.fill_between(x_axis, dpp_reward + dpp_stdev, dpp_reward - dpp_stdev, alpha=alpha_val, facecolor=color3)

    else:
        print("INCORRECT REWARD TYPE")

    # Graph Details
    plt.xlabel("Generations")
    plt.ylabel("Global Reward")
    plt.legend(["Rover Policy Rewards"])

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/Rover_Policy_Learning_Curve.pdf")
    plt.close()


def generate_rover_heatmap(sruns, n_rovers, n_poi):
    # Plot Colors
    blue = np.array([26, 100, 255]) / 255
    orange = np.array([255, 130, 0]) / 255
    m_size = 0.001
    n_colors = int(1 / m_size)
    hcolors = [orange, blue]
    hmap_colors = LinearSegmentedColormap.from_list("Custom", hcolors, N=n_colors)

    # File Paths
    r1_path = '../Output_Data/Rover0POIVisits.csv'
    r2_path = '../Output_Data/Rover1POIVisits.csv'
    r3_path = '../Output_Data/Rover2POIVisits.csv'
    r4_path = '../Output_Data/Rover3POIVisits.csv'
    r5_path = '../Output_Data/Rover4POIVisits.csv'

    i = 0
    rover_data = np.zeros((n_rovers, n_poi + 1))
    rov_input1 = []
    with open(r1_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input1.append(row)
    for row in rov_input1:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1

    rov_input2 = []
    with open(r2_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input2.append(row)
    i += 1
    for row in rov_input2:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1

    rov_input3 = []
    with open(r3_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input3.append(row)
    i += 1
    for row in rov_input3:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1

    rov_input4 = []
    with open(r4_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input4.append(row)
    i += 1
    for row in rov_input4:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1

    # rov_input5 = []
    # with open(r5_path) as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',')
    #     for row in csv_reader:
    #         rov_input5.append(row)
    # i += 1
    # for row in rov_input5:
    #     j = 0
    #     for val in row:
    #         rover_data[i, j] += float(val)
    #         j += 1

    rover_data /= sruns

    # Create The Plot
    x_axis = [0, 1, 2, 3]
    # x_axis = [0, 1, 2, 3, 4]
    # y_axis = [0, 1]
    y_axis = [0, 1, 2, 3]
    # y_axis = [0, 1, 2, 3, 4]

    hmap_data = rover_data[:, 0:n_poi].copy()
    plt.imshow(hmap_data.T, cmap="Oranges")
    plt.xticks(x_axis)
    plt.yticks(y_axis)
    plt.xlabel("Agent ID")
    plt.ylabel("POI Observed")
    plt.colorbar()

    # Add gridlines (requires shifting major/mior gridlines)
    ax = plt.gca()
    # Major ticks
    ax.set_xticks(np.arange(0, len(x_axis), 1))
    ax.set_yticks(np.arange(0, len(y_axis), 1))

    # Labels for major ticks
    ax.set_xticklabels(['A1', 'A2', 'A3', 'A4'])
    # ax.set_xticklabels(['A1', 'A2', 'A3', 'A4', 'A5'])
    # ax.set_yticklabels(['P1', 'P2'])
    ax.set_yticklabels(['P1', 'P2', 'P3', 'P4'])
    # ax.set_yticklabels(['P1', 'P2', 'P3', 'P4', 'P5'])

    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(x_axis), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(y_axis), 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)

    # Mark the Hazardous POI
    # ax.get_yticklabels()[0].set_color('blue')
    # ax.get_yticklabels()[0].set_fontweight('bold')
    # ax.get_yticklabels()[1].set_color('blue')
    # ax.get_yticklabels()[1].set_fontweight('bold')
    ax.get_yticklabels()[3].set_color('blue')
    ax.get_yticklabels()[3].set_fontweight('bold')

    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig(f'Plots/P{n_poi}R{n_rovers}_Heatmap.pdf')
    plt.close()


def generate_supervisor_heatmap(sruns, n_rovers, n_poi):
    # Plot Colors
    blue = np.array([26, 100, 255]) / 255
    orange = np.array([255, 130, 0]) / 255
    m_size = 0.001
    n_colors = int(1 / m_size)
    hcolors = [orange, blue]
    hmap_colors = LinearSegmentedColormap.from_list("Custom", hcolors, N=n_colors)

    # File Paths
    r1_path = '../Output_Data/ACGRover0POIVisits.csv'
    r2_path = '../Output_Data/ACGRover1POIVisits.csv'
    r3_path = '../Output_Data/ACGRover2POIVisits.csv'
    r4_path = '../Output_Data/ACGRover3POIVisits.csv'
    r5_path = '../Output_Data/ACGRover4POIVisits.csv'

    i = 0
    rover_data = np.zeros((n_rovers, n_poi + 1))
    rov_input1 = []
    with open(r1_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input1.append(row)
    for row in rov_input1:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1

    rov_input2 = []
    with open(r2_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input2.append(row)
    i += 1
    for row in rov_input2:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1

    rov_input3 = []
    with open(r3_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input3.append(row)
    i += 1
    for row in rov_input3:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1

    rov_input4 = []
    with open(r4_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input4.append(row)
    i += 1
    for row in rov_input4:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1

    # rov_input5 = []
    # with open(r5_path) as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',')
    #     for row in csv_reader:
    #         rov_input5.append(row)
    # i += 1
    # for row in rov_input5:
    #     j = 0
    #     for val in row:
    #         rover_data[i, j] += float(val)
    #         j += 1

    rover_data /= sruns

    # Create The Plot
    x_axis = [0, 1, 2, 3]
    # x_axis = [0, 1, 2, 3, 4]
    # y_axis = [0, 1]
    y_axis = [0, 1, 2, 3]
    # y_axis = [0, 1, 2, 3, 4]

    hmap_data = rover_data[:, 0:n_poi].copy()
    plt.imshow(hmap_data.T, cmap='Oranges')
    plt.xticks(x_axis)
    plt.yticks(y_axis)
    plt.xlabel("Agent ID")
    plt.ylabel("POI Observed")
    plt.colorbar()

    # Add gridlines (requires shifting major/mior gridlines)
    ax = plt.gca()
    # Major ticks
    ax.set_xticks(np.arange(0, len(x_axis), 1))
    ax.set_yticks(np.arange(0, len(y_axis), 1))

    # Labels for major ticks
    ax.set_xticklabels(['A1', 'A2', 'A3', 'A4'])
    # ax.set_xticklabels(['A1', 'A2', 'A3', 'A4', 'A5'])
    # ax.set_yticklabels(['P1', 'P2'])
    ax.set_yticklabels(['P1', 'P2', 'P3', 'P4'])
    # ax.set_yticklabels(['P1', 'P2', 'P3', 'P4', 'P5'])

    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(x_axis), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(y_axis), 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)

    # Mark the Hazardous POI
    # ax.get_yticklabels()[0].set_color('blue')
    # ax.get_yticklabels()[0].set_fontweight('bold')
    # ax.get_yticklabels()[1].set_color('blue')
    # ax.get_yticklabels()[1].set_fontweight('bold')
    ax.get_yticklabels()[3].set_color('blue')
    ax.get_yticklabels()[3].set_fontweight('bold')

    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig(f'Plots/P{n_poi}R{n_rovers}_ACGHeatmap.pdf')
    plt.close()


if __name__ == '__main__':
    """
    This code generates the training plots for training the counterfactual supervisor.
    When calling script, reward type that agents are trained with needs to be entered (Global, D, or D++)
    """
    n_rovers = 4
    n_poi = 4
    generations = 500
    acg_generations = 2000
    sruns = 30
    reward_type = sys.argv[1]
    sample_rate = 20

    generate_policy_learning_curves(generations, sample_rate, sruns, reward_type)
    generate_acg_learning_curves(acg_generations, sample_rate, sruns)
    generate_rover_heatmap(sruns, n_rovers, n_poi)
    generate_supervisor_heatmap(sruns, n_rovers, n_poi)
