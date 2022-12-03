import matplotlib.pyplot as plt
import numpy as np
import csv
from parameters import parameters as p
import math


def import_reward_data(csv_name):

    config_input = []
    with open(csv_name) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        for row in csv_reader:
            config_input.append(row)

    n_data_points = int((p["generations"]/p["sample_rate"])+1)
    average_reward = np.zeros(n_data_points)
    for row in config_input:
        for i in range(n_data_points):
            average_reward[i] += float(row[i])

    average_reward /= p["stat_runs"]

    return average_reward


def get_standard_deviations(csv_name, data_mean):
    config_input = []
    with open(csv_name) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        for row in csv_reader:
            config_input.append(row)

    n_data_points = int((p["generations"] / p["sample_rate"]) + 1)
    assert(len(data_mean) == n_data_points)

    temp_array = np.zeros(n_data_points)
    standard_dev = np.zeros(n_data_points)
    for row in config_input:
        for i in range(n_data_points):
            temp_array[i] += (float(row[i]) - data_mean[i])**2

    temp_array /= p["stat_runs"]
    for i in range(n_data_points):
        standard_dev[i] = math.sqrt(temp_array[i])/math.sqrt(p["stat_runs"])

    return standard_dev


def generate_cfl_learning_curves():
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia

    g_reward = import_reward_data('../Output_Data/Global_Reward.csv')
    g_stdev = get_standard_deviations('../Output_Data/Global_Reward.csv', g_reward)
    d_reward = import_reward_data('../Output_Data/Difference_Reward.csv')
    d_stdev = get_standard_deviations('../Output_Data/Difference_Reward.csv', d_reward)
    dpp_reward = import_reward_data('../Output_Data/DPP_Reward.csv')
    dpp_stdev = get_standard_deviations('../Output_Data/DPP_Reward.csv', dpp_reward)
    high_cfl_reward = import_reward_data('../Output_Data/SDPP_Reward.csv')
    high_cfl_stdev = get_standard_deviations('../Output_Data/SDPP_Reward.csv', high_cfl_reward)
    # low_cfl_reward = import_reward_data('../Output_Data/SDPP_Reward.csv')
    # low_cfl_stdev = get_standard_deviations('../Output_Data/SDPP_Reward.csv', low_cfl_reward)

    x_axis = []
    for i in range(p["generations"]):
        if i % p["sample_rate"] == 0 or i == p["generations"]-1:
            x_axis.append(i)
    x_axis = np.array(x_axis)

    # Plot of Data
    plt.plot(x_axis, g_reward, color=color1)
    plt.plot(x_axis, d_reward, color=color2)
    plt.plot(x_axis, dpp_reward, color=color3)
    plt.plot(x_axis, high_cfl_reward, color=color4)
    # plt.plot(x_axis, low_cfl_reward, color=color5)

    # Plot of Error
    alpha_val = 0.2
    plt.fill_between(x_axis, g_reward + g_stdev, g_reward - g_stdev, alpha=alpha_val, facecolor=color1)
    plt.fill_between(x_axis, d_reward + d_stdev, d_reward - d_stdev, alpha=alpha_val, facecolor=color2)
    plt.fill_between(x_axis, dpp_reward + dpp_stdev, dpp_reward - dpp_stdev, alpha=alpha_val, facecolor=color3)
    plt.fill_between(x_axis, high_cfl_reward + high_cfl_stdev, high_cfl_reward - high_cfl_stdev, alpha=alpha_val, facecolor=color4)
    # plt.fill_between(x_axis, low_cfl_reward + low_cfl_stdev, low_cfl_reward - low_cfl_stdev, alpha=alpha_val, facecolor=color5)

    # Graph Details
    plt.xlabel("Generations")
    plt.ylabel("Global Reward")
    plt.legend(["Global", "Difference", "D++", "CFL"])
    plt.show()


if __name__ == '__main__':
    generate_cfl_learning_curves()
