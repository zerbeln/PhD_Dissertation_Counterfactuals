import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from plots_common_functions import import_reward_data, get_standard_deviations


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
    acg_stdev = get_standard_deviations(acg_file_path, acg_reward, generations, sample_rate, sruns)

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
        g_stdev = get_standard_deviations(g_file_path, g_reward, generations, sample_rate, sruns)

        # Plot of Data
        plt.plot(x_axis, g_reward, color=color1)
        # Plot of Error
        plt.fill_between(x_axis, g_reward + g_stdev, g_reward - g_stdev, alpha=alpha_val, facecolor=color1)

    elif reward_type == "Difference":
        d_file_path = '../Output_Data/Difference_Reward.csv'
        d_reward = import_reward_data(d_file_path, generations, sample_rate, sruns)
        d_stdev = get_standard_deviations(d_file_path, d_reward, generations, sample_rate, sruns)

        # Plot of Data
        plt.plot(x_axis, d_reward, color=color2)
        # Plot of Error
        plt.fill_between(x_axis, d_reward + d_stdev, d_reward - d_stdev, alpha=alpha_val, facecolor=color2)

    elif reward_type == "DPP":
        dpp_file_path = '../Output_Data/DPP_Reward.csv'
        dpp_reward = import_reward_data(dpp_file_path, generations, sample_rate, sruns)
        dpp_stdev = get_standard_deviations(dpp_file_path, dpp_reward, generations, sample_rate, sruns)

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


if __name__ == '__main__':
    generations = 500
    acg_generations = 1000
    sruns = 15
    reward_type = sys.argv[1]
    sample_rate = 20

    generate_policy_learning_curves(generations, sample_rate, sruns, reward_type)
    generate_acg_learning_curves(acg_generations, sample_rate, sruns)
