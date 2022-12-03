import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import sys
import os


def import_reward_data(csv_name, generations, sample_rate, sruns):

    config_input = []
    with open(csv_name) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        for row in csv_reader:
            config_input.append(row)

    n_data_points = int((generations/sample_rate)+1)
    average_reward = np.zeros(n_data_points)
    for row in config_input:
        for i in range(n_data_points):
            average_reward[i] += float(row[i])

    average_reward /= sruns

    return average_reward


def get_standard_deviations(csv_name, data_mean, generations, sample_rate, sruns):
    config_input = []
    with open(csv_name) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        for row in csv_reader:
            config_input.append(row)

    n_data_points = int((generations/sample_rate) + 1)
    assert(len(data_mean) == n_data_points)

    temp_array = np.zeros(n_data_points)
    standard_dev = np.zeros(n_data_points)
    for row in config_input:
        for i in range(n_data_points):
            temp_array[i] += (float(row[i]) - data_mean[i])**2

    temp_array /= sruns
    for i in range(n_data_points):
        standard_dev[i] = math.sqrt(temp_array[i])/math.sqrt(sruns)

    return standard_dev


def generate_cfl_learning_curves(generations, sample_rate, sruns):

    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    # Graph Data
    g_file_path ='../Global/Output_Data/Global_Reward.csv'
    g_reward = import_reward_data(g_file_path, generations, sample_rate, sruns)
    g_stdev = get_standard_deviations(g_file_path, g_reward, generations, sample_rate, sruns)

    d_file_path = '../Difference/Output_Data/Difference_Reward.csv'
    d_reward = import_reward_data(d_file_path, generations, sample_rate, sruns)
    d_stdev = get_standard_deviations(d_file_path, d_reward, generations, sample_rate, sruns)

    dpp_file_path = '../D++/Output_Data/DPP_Reward.csv'
    dpp_reward = import_reward_data(dpp_file_path, generations, sample_rate, sruns)
    dpp_stdev = get_standard_deviations(dpp_file_path, dpp_reward, generations, sample_rate, sruns)

    high_cfl_file_path = '../High_SD++/Output_Data/SDPP_Reward.csv'
    high_cfl_reward = import_reward_data(high_cfl_file_path, generations, sample_rate, sruns)
    high_cfl_stdev = get_standard_deviations(high_cfl_file_path, high_cfl_reward, generations, sample_rate, sruns)

    low_cfl_file_path = '../Low_SD++/Output_Data/SDPP_Reward.csv'
    low_cfl_reward = import_reward_data(low_cfl_file_path, generations, sample_rate, sruns)
    low_cfl_stdev = get_standard_deviations(low_cfl_file_path, low_cfl_reward, generations, sample_rate, sruns)

    x_axis = []
    for i in range(generations):
        if i % sample_rate == 0 or i == generations-1:
            x_axis.append(i)
    x_axis = np.array(x_axis)

    # Plot of Data
    plt.plot(x_axis, g_reward, color=color1)
    plt.plot(x_axis, d_reward, color=color2)
    plt.plot(x_axis, dpp_reward, color=color3)
    plt.plot(x_axis, high_cfl_reward, color=color4)
    plt.plot(x_axis, low_cfl_reward, color=color5)

    # Plot of Error
    alpha_val = 0.2
    plt.fill_between(x_axis, g_reward + g_stdev, g_reward - g_stdev, alpha=alpha_val, facecolor=color1)
    plt.fill_between(x_axis, d_reward + d_stdev, d_reward - d_stdev, alpha=alpha_val, facecolor=color2)
    plt.fill_between(x_axis, dpp_reward + dpp_stdev, dpp_reward - dpp_stdev, alpha=alpha_val, facecolor=color3)
    plt.fill_between(x_axis, high_cfl_reward + high_cfl_stdev, high_cfl_reward - high_cfl_stdev, alpha=alpha_val, facecolor=color4)
    plt.fill_between(x_axis, low_cfl_reward + low_cfl_stdev, low_cfl_reward - low_cfl_stdev, alpha=alpha_val, facecolor=color5)

    # Graph Details
    plt.xlabel("Generations")
    plt.ylabel("Global Reward")
    plt.legend(["Global", "Difference", "D++", "CFL Max", "CFL Min"])

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/CFL_Learning_Curve.pdf")

    # Show the plot
    plt.show()


if __name__ == '__main__':
    generations = int(sys.argv[1])
    sample_rate = int(sys.argv[2])
    sruns = int(sys.argv[3])

    generate_cfl_learning_curves(generations, sample_rate, sruns)
