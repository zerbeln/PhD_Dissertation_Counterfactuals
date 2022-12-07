import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import sys
import os
from plots_common_functions import import_reward_data, get_standard_deviations


def create_coupling_plots(max_coupling, n_poi, n_rovers):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia
    x_axis = ["C1", "C2", "C3", "C4", "C5", "C6"]

    g_data = []
    d_data = []
    dpp_data = []
    cba_data = []

    for i in range(max_coupling):
        g_path = '../Coupling/C{0}/Global/Output_Data/Final_GlobalRewards.csv'.format(i+1)
        d_path = '../Coupling/C{0}/Difference/Output_Data/Final_GlobalRewards.csv'.format(i+1)
        dpp_path = '../Coupling/C{0}/D++/Output_Data/Final_GlobalRewards.csv'.format(i+1)
        cba_path = '../Coupling/C{0}/CBA/Output_Data/Final_GlobalRewards.csv'.format(i+1)

        g_input = []
        d_input = []
        dpp_input = []
        cba_input = []

        with open(g_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                g_input.append(row)

        temp = []
        for row in g_input:
            for val in row:
                temp.append(float(val))
        g_data.append(np.mean(temp))

        with open(d_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                d_input.append(row)
        temp = []
        for row in d_input:
            for val in row:
                temp.append(float(val))
        d_data.append(np.mean(temp))

        with open(dpp_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                dpp_input.append(row)
        temp = []
        for row in dpp_input:
            for val in row:
                temp.append(float(val))
        dpp_data.append(np.mean(temp))

        with open(cba_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                cba_input.append(row)
        temp = []
        for row in cba_input:
            for val in row:
                temp.append(float(val))
        cba_data.append(np.mean(temp))

    plt.plot(x_axis, g_data, color=color1, linestyle='--', marker='^')
    plt.plot(x_axis, d_data, color=color2, linestyle='-.', marker='o')
    plt.plot(x_axis, dpp_data, color=color3, linestyle=':', marker='s')
    plt.plot(x_axis, cba_data, color=color4, marker='H')

    plt.xlabel("POI Coupling Requirement")
    plt.ylabel("Percentage of Maximum Score")
    plt.legend(["G", "D", "D++", "CBA"])

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/P{0}R{1}_Coupling.pdf".format(n_poi, n_rovers))

    plt.show()


def create_hazard_performance_plots(n_poi, n_rovers):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia
    colors = [color1, color2, color4, color3, color5]
    x_axis = ["G", "D", "CBA-C1", "D++", "CBA-C3"]

    g_data = []
    d_data = []
    dpp_data = []
    cba_c1_data = []
    cba_c3_data = []

    for i in range(3):
        g_path = '../Hazards/H{0}/Global/Output_Data/Final_GlobalRewards.csv'.format(i + 1)
        d_path = '../Hazards/H{0}/Difference/Output_Data/Final_GlobalRewards.csv'.format(i + 1)
        dpp_path = '../Hazards/H{0}/D++/Output_Data/Final_GlobalRewards.csv'.format(i + 1)
        cba_c1_path = '../Hazards/H{0}/CBA_LC/Output_Data/Final_GlobalRewards.csv'.format(i + 1)
        cba_c3_path = '../Hazards/H{0}/CBA_TC/Output_Data/Final_GlobalRewards.csv'.format(i + 1)

        g_input = []
        d_input = []
        dpp_input = []
        cba_c1_input = []
        cba_c3_input = []

        with open(g_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                g_input.append(row)
        temp = []
        for row in g_input:
            for val in row:
                temp.append(float(val))
        g_data.append(np.mean(temp))

        with open(d_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                d_input.append(row)
        temp = []
        for row in d_input:
            for val in row:
                temp.append(float(val))
        d_data.append(np.mean(temp))

        with open(dpp_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                dpp_input.append(row)
        temp = []
        for row in dpp_input:
            for val in row:
                temp.append(float(val))
        dpp_data.append(np.mean(temp))

        with open(cba_c1_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                cba_c1_input.append(row)
        temp = []
        for row in cba_c1_input:
            for val in row:
                temp.append(float(val))
        cba_c1_data.append(np.mean(temp))

        with open(cba_c3_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                cba_c3_input.append(row)
        temp = []
        for row in cba_c3_input:
            for val in row:
                temp.append(float(val))
        cba_c3_data.append(np.mean(temp))

    h1_ydata = [g_data[0], d_data[0], cba_c1_data[0], dpp_data[0], cba_c3_data[0]]
    h2_ydata = [g_data[1], d_data[1], cba_c1_data[1], dpp_data[1], cba_c3_data[1]]
    h3_ydata = [g_data[2], d_data[2], cba_c1_data[2], dpp_data[2], cba_c3_data[2]]

    fig, (h1, h2, h3) = plt.subplots(nrows=1, ncols=3)

    h1.bar(x_axis, h1_ydata, color=colors)
    h1.set_ylabel("Average Team Reward")
    h1.set_title("3 Hazards")
    h1.tick_params('x', labelrotation=45)

    h2.bar(x_axis, h2_ydata, color=colors)
    h2.set_title("4 Hazards")
    h2.tick_params('x', labelrotation=45)

    h3.bar(x_axis, h3_ydata, color=colors)
    h3.set_title("6 Hazards")
    h3.tick_params('x', labelrotation=45)

    fig.tight_layout()
    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/P{0}R{1}_Hazards.pdf".format(n_poi, n_rovers))
    plt.show()

def create_hazard_incursion_plots(n_poi, n_rovers):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia
    colors = [color1, color2, color4, color3, color5]
    x_axis = ["G", "D", "CBA-C1", "D++", "CBA-C3"]

    g_data = []
    d_data = []
    dpp_data = []
    cba_c1_data = []
    cba_c3_data = []

    for i in range(3):
        g_path = '../Hazards/H{0}/Global/Output_Data/HazardIncursions.csv'.format(i + 1)
        d_path = '../Hazards/H{0}/Difference/Output_Data/HazardIncursions.csv'.format(i + 1)
        dpp_path = '../Hazards/H{0}/D++/Output_Data/HazardIncursions.csv'.format(i + 1)
        cba_c1_path = '../Hazards/H{0}/CBA_LC/Output_Data/HazardIncursions.csv'.format(i + 1)
        cba_c3_path = '../Hazards/H{0}/CBA_TC/Output_Data/HazardIncursions.csv'.format(i + 1)

        g_input = []
        d_input = []
        dpp_input = []
        cba_c1_input = []
        cba_c3_input = []

        with open(g_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                g_input.append(row)
        temp = []
        for row in g_input:
            for val in row:
                temp.append(float(val))
        g_data.append(np.mean(temp))

        with open(d_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                d_input.append(row)
        temp = []
        for row in d_input:
            for val in row:
                temp.append(float(val))
        d_data.append(np.mean(temp))

        with open(dpp_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                dpp_input.append(row)
        temp = []
        for row in dpp_input:
            for val in row:
                temp.append(float(val))
        dpp_data.append(np.mean(temp))

        with open(cba_c1_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                cba_c1_input.append(row)
        temp = []
        for row in cba_c1_input:
            for val in row:
                temp.append(float(val))
        cba_c1_data.append(np.mean(temp))

        with open(cba_c3_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                cba_c3_input.append(row)
        temp = []
        for row in cba_c3_input:
            for val in row:
                temp.append(float(val))
        cba_c3_data.append(np.mean(temp))

    h1_ydata = [g_data[0], d_data[0], cba_c1_data[0], dpp_data[0], cba_c3_data[0]]
    h2_ydata = [g_data[1], d_data[1], cba_c1_data[1], dpp_data[1], cba_c3_data[1]]
    h3_ydata = [g_data[2], d_data[2], cba_c1_data[2], dpp_data[2], cba_c3_data[2]]

    fig, (h1, h2, h3) = plt.subplots(nrows=1, ncols=3)

    h1.bar(x_axis, h1_ydata, color=colors)
    h1.set_ylabel("Number of Rover Incursions")
    h1.set_title("3 Hazards")
    h1.tick_params('x', labelrotation=45)

    h2.bar(x_axis, h2_ydata, color=colors)
    h2.set_title("4 Hazards")
    h2.tick_params('x', labelrotation=45)

    h3.bar(x_axis, h3_ydata, color=colors)
    h3.set_title("6 Hazards")
    h3.tick_params('x', labelrotation=45)

    fig.tight_layout()
    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/P{0}R{1}_Incursions.pdf".format(n_poi, n_rovers))
    plt.show()


if __name__ == '__main__':
    # generations = int(sys.argv[1])
    # sample_rate = int(sys.argv[2])
    # sruns = int(sys.argv[3])

    generations = 5000
    sample_rate = 20
    sruns = 30

    max_coupling = 6
    n_rovers = 6
    n_poi =5

    create_coupling_plots(max_coupling, n_poi, n_rovers)
    create_hazard_performance_plots(10, 6)
    create_hazard_incursion_plots(10, 6)
