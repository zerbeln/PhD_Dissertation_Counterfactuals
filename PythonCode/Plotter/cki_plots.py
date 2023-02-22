import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import csv
import sys
import os
from plots_common_functions import import_reward_data, get_standard_err_performance


def calc_max_poi_val(n_rovers, poi_coupling):
    poi_path = '../C{0}/CKI/World_Config/POI_Config.csv'.format(poi_coupling)

    config_input = []
    with open(poi_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        for row in csv_reader:
            config_input.append(row)

    poi_vals = []
    n_poi = len(config_input)
    for poi_id in range(n_poi):
        poi_vals.append(float(config_input[poi_id][2]))

    max_val = 0
    poi_vals = np.sort(poi_vals)[::-1]
    n_possible = int(n_rovers/poi_coupling)
    if n_possible >= n_poi:
        max_val = sum(poi_vals)
    else:
        for i in range(n_possible):
            max_val += poi_vals[i]

    return max_val


def create_coupling_plots(max_coupling, n_poi, n_rovers, sruns):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia
    x_axis = ["C1", "C2", "C3", "C4", "C5", "C6"]

    g_data = []
    g_err = []
    d_data = []
    d_err = []
    dpp_data = []
    dpp_err = []
    cki_data = []
    cki_err = []

    for i in range(max_coupling):
        g_path = '../C{0}/Global/Output_Data/Final_GlobalRewards.csv'.format(i+1)
        d_path = '../C{0}/Difference/Output_Data/Final_GlobalRewards.csv'.format(i+1)
        dpp_path = '../C{0}/D++/Output_Data/Final_GlobalRewards.csv'.format(i+1)
        cki_path = '../C{0}/CKI/Output_Data/Final_GlobalRewards.csv'.format(i+1)

        max_reward = calc_max_poi_val(n_rovers, i+1)  # Maximum possible score teams can achieve
        g_input = []
        d_input = []
        dpp_input = []
        cki_input = []

        # Global Reward Performance ----------------------------
        with open(g_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                g_input.append(row)
        temp = []
        for row in g_input:
            for val in row:
                temp.append(float(val))
        g_data.append(np.mean(temp)/max_reward)  # Scaled with respect to maximum possible reward
        g_err.append(get_standard_err_performance(g_path, np.mean(temp), sruns)/max_reward)

        # Difference Reward Performance --------------------------
        with open(d_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                d_input.append(row)
        temp = []
        for row in d_input:
            for val in row:
                temp.append(float(val))
        d_data.append(np.mean(temp)/max_reward)  # Scaled with respect to maximum possible reward
        d_err.append(get_standard_err_performance(d_path, np.mean(temp), sruns)/max_reward)

        # D++ Reward Performance ------------------------------
        with open(dpp_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                dpp_input.append(row)
        temp = []
        for row in dpp_input:
            for val in row:
                temp.append(float(val))
        dpp_data.append(np.mean(temp)/max_reward)  # Scaled with respect to maximum possible reward
        dpp_err.append(get_standard_err_performance(dpp_path, np.mean(temp), sruns)/max_reward)

        # CKI Performance -------------------------------------
        with open(cki_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                cki_input.append(row)
        temp = []
        for row in cki_input:
            for val in row:
                temp.append(float(val))
        cki_data.append(np.mean(temp)/max_reward)  # Scaled with respect to maximum possible reward
        cki_err.append(get_standard_err_performance(cki_path, np.mean(temp), sruns)/max_reward)

    # POI Value Estimates may not be exact if rovers can reach multiple POI
    for i in range(len(dpp_data)):
        if g_data[i] > 1.0:
            g_data[i] = 1.0
        if d_data[i] > 1.0:
            d_data[i] = 1.0
        if dpp_data[i] > 1.0:
            dpp_data[i] = 1.0
        if cki_data[i] > 1.0:
            cki_data[i] = 1.0

    plt.errorbar(x_axis, g_data, g_err, color=color3, linestyle='--', marker='^')
    plt.errorbar(x_axis, d_data, d_err, color=color2, linestyle='-.', marker='o')
    plt.errorbar(x_axis, dpp_data, dpp_err, color=color1, linestyle=':', marker='s')
    plt.errorbar(x_axis, cki_data, cki_err, color=color4, marker='H')

    plt.xlabel("POI Coupling Requirement")
    plt.ylabel("Percentage of Maximum Fitness")
    plt.legend(["G", "D", "D++", "CKI"])

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/P{0}R{1}_Coupling.pdf".format(n_poi, n_rovers))
    plt.close()


def create_hazard_performance_plots(n_poi, n_rovers, sruns):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia
    colors = [color3, color2, color4, color1, color5]
    x_axis = ["G", "D", "CKI-C1", "D++", "CKI-C3"]

    g_data = []
    g_err = []
    d_data = []
    d_err = []
    dpp_data = []
    dpp_err = []
    cki_c1_data = []
    cki_c1_err = []
    cki_c3_data = []
    cki_c3_err = []

    for i in range(3):
        g_path = '../H{0}/Global/Output_Data/Final_GlobalRewards.csv'.format(i + 1)
        d_path = '../H{0}/Difference/Output_Data/Final_GlobalRewards.csv'.format(i + 1)
        dpp_path = '../H{0}/D++/Output_Data/Final_GlobalRewards.csv'.format(i + 1)
        cki_c1_path = '../H{0}/CBA_LC/Output_Data/Final_GlobalRewards.csv'.format(i + 1)
        cki_c3_path = '../H{0}/CBA_TC/Output_Data/Final_GlobalRewards.csv'.format(i + 1)

        g_input = []
        d_input = []
        dpp_input = []
        cki_c1_input = []
        cki_c3_input = []

        # Global Reward Performance ----------------------------------
        with open(g_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                g_input.append(row)
        temp = []
        for row in g_input:
            for val in row:
                temp.append(float(val))
        g_data.append(np.mean(temp))
        g_err.append(get_standard_err_performance(g_path, np.mean(temp), sruns))

        # Difference Reward Performance -----------------------------
        with open(d_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                d_input.append(row)
        temp = []
        for row in d_input:
            for val in row:
                temp.append(float(val))
        d_data.append(np.mean(temp))
        d_err.append(get_standard_err_performance(d_path, np.mean(temp), sruns))

        # D++ Reward Performance ---------------------------------
        with open(dpp_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                dpp_input.append(row)
        temp = []
        for row in dpp_input:
            for val in row:
                temp.append(float(val))
        dpp_data.append(np.mean(temp))
        dpp_err.append(get_standard_err_performance(dpp_path, np.mean(temp), sruns))

        # CKI-1 Performance ----------------------------------
        with open(cki_c1_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                cki_c1_input.append(row)
        temp = []
        for row in cki_c1_input:
            for val in row:
                temp.append(float(val))
        cki_c1_data.append(np.mean(temp))
        cki_c1_err.append(get_standard_err_performance(cki_c1_path, np.mean(temp), sruns))

        # CKI-3 Performance ----------------------------------
        with open(cki_c3_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                cki_c3_input.append(row)
        temp = []
        for row in cki_c3_input:
            for val in row:
                temp.append(float(val))
        cki_c3_data.append(np.mean(temp))
        cki_c3_err.append(get_standard_err_performance(cki_c3_path, np.mean(temp), sruns))

    h1_ydata = [g_data[0], d_data[0], cki_c1_data[0], dpp_data[0], cki_c3_data[0]] + np.array([1050.0 for i in range(5)])
    h1_err = [g_err[0], d_err[0], cki_c1_err[0], dpp_err[0], cki_c3_err[0]]
    h2_ydata = [g_data[1], d_data[1], cki_c1_data[1], dpp_data[1], cki_c3_data[1]] + np.array([1050.0 for i in range(5)])
    h2_err = [g_err[1], d_err[1], cki_c1_err[1], dpp_err[1], cki_c3_err[1]]
    h3_ydata = [g_data[2], d_data[2], cki_c1_data[2], dpp_data[2], cki_c3_data[2]] + np.array([1050.0 for i in range(5)])
    h3_err = [g_err[2], d_err[2], cki_c1_err[2], dpp_err[2], cki_c3_err[2]]

    fig, (h1, h2, h3) = plt.subplots(nrows=1, ncols=3)

    h1.bar(x_axis, h1_ydata, yerr=h1_err, color=colors)
    h1.axhline(0, color='black')
    h1.axvline(2.5, color='k', linestyle="--")
    h1.set_ylabel("Average Team Fitness")
    h1.set_title("3 Hazards")
    h1.tick_params('x', labelrotation=45)
    h1.set_yticks(np.arange(0, 1105, 100))
    h1.set_ylim([0, 1100])

    h2.bar(x_axis, h2_ydata, yerr=h2_err, color=colors)
    h2.axhline(0, color='black')
    h2.axvline(2.5, color='k', linestyle="--")
    h2.set_title("4 Hazards")
    h2.tick_params('x', labelrotation=45)
    h2.set_yticks(np.arange(0, 1105, 100))
    h2.set_ylim([0, 1100])

    h3.bar(x_axis, h3_ydata, yerr=h3_err, color=colors)
    h3.axhline(0, color='black')
    h3.axvline(2.5, color='k', linestyle="--")
    h3.set_title("6 Hazards")
    h3.tick_params('x', labelrotation=45)
    h3.set_yticks(np.arange(0, 1105, 100))
    h3.set_ylim([0, 1100])

    fig.tight_layout()
    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/P{0}R{1}_Hazards.pdf".format(n_poi, n_rovers))
    plt.close()


def create_hazard_incursion_plots(n_poi, n_rovers, sruns):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia
    colors = [color3, color2, color4, color1, color5]
    x_axis = ["G", "D", "CKI-C1", "D++", "CKI-C3"]

    g_data = []
    g_err = []
    d_data = []
    d_err = []
    dpp_data = []
    dpp_err = []
    cki_c1_data = []
    cki_c1_err = []
    cki_c3_data = []
    cki_c3_err = []

    for i in range(3):
        g_path = '../H{0}/Global/Output_Data/HazardIncursions.csv'.format(i + 1)
        d_path = '../H{0}/Difference/Output_Data/HazardIncursions.csv'.format(i + 1)
        dpp_path = '../H{0}/D++/Output_Data/HazardIncursions.csv'.format(i + 1)
        cki_c1_path = '../H{0}/CBA_LC/Output_Data/HazardIncursions.csv'.format(i + 1)
        cki_c3_path = '../H{0}/CBA_TC/Output_Data/HazardIncursions.csv'.format(i + 1)

        g_input = []
        d_input = []
        dpp_input = []
        cki_c1_input = []
        cki_c3_input = []

        # Global Reward Incursions ----------------------------
        with open(g_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                g_input.append(row)
        temp = []
        for row in g_input:
            for val in row:
                temp.append(float(val))
        g_data.append(np.mean(temp))
        g_err.append(get_standard_err_performance(g_path, np.mean(temp), sruns))

        # Difference Reward Incursions ------------------------
        with open(d_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                d_input.append(row)
        temp = []
        for row in d_input:
            for val in row:
                temp.append(float(val))
        d_data.append(np.mean(temp))
        d_err.append(get_standard_err_performance(d_path, np.mean(temp), sruns))

        # D++ Reward Incursions ------------------------------
        with open(dpp_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                dpp_input.append(row)
        temp = []
        for row in dpp_input:
            for val in row:
                temp.append(float(val))
        dpp_data.append(np.mean(temp))
        dpp_err.append(get_standard_err_performance(dpp_path, np.mean(temp), sruns))

        # CKI-1 Incursions -----------------------------------
        with open(cki_c1_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                cki_c1_input.append(row)
        temp = []
        for row in cki_c1_input:
            for val in row:
                temp.append(float(val))
        cki_c1_data.append(np.mean(temp))
        cki_c1_err.append(get_standard_err_performance(cki_c1_path, np.mean(temp), sruns))

        # CKI-3 Incursions ----------------------------------
        with open(cki_c3_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                cki_c3_input.append(row)
        temp = []
        for row in cki_c3_input:
            for val in row:
                temp.append(float(val))
        cki_c3_data.append(np.mean(temp))
        cki_c3_err.append(get_standard_err_performance(cki_c3_path, np.mean(temp), sruns))

    # Results are scaled for better performance comparisons
    h1_ydata = [g_data[0], d_data[0], cki_c1_data[0], dpp_data[0], cki_c3_data[0]]
    h1_err = [g_err[0], d_err[0], cki_c1_err[0], dpp_err[0], cki_c3_err[0]]
    h2_ydata = [g_data[1], d_data[1], cki_c1_data[1], dpp_data[1], cki_c3_data[1]]
    h2_err = [g_err[1], d_err[1], cki_c1_err[1], dpp_err[1], cki_c3_err[1]]
    h3_ydata = [g_data[2], d_data[2], cki_c1_data[2], dpp_data[2], cki_c3_data[2]]
    h3_err = [g_err[2], d_err[2], cki_c1_err[2], dpp_err[2], cki_c3_err[2]]

    fig, (h1, h2, h3) = plt.subplots(nrows=3, ncols=1)

    h1.barh(x_axis, h1_ydata, xerr=h1_err, color=colors)
    h1.axhline(2.5, color='k', linestyle='--')
    h1.set_title("3 Hazards")
    h1.tick_params('x')
    h1.invert_yaxis()

    h2.barh(x_axis, h2_ydata, xerr=h2_err, color=colors)
    h2.axhline(2.5, color='k', linestyle='--')
    h2.set_title("4 Hazards")
    h2.tick_params('x')
    h2.invert_yaxis()

    h3.barh(x_axis, h3_ydata, xerr=h3_err, color=colors)
    h3.axhline(2.5, color='k', linestyle='--')
    h3.set_title("6 Hazards")
    h3.set_xlabel("Number of Hazard Incursions")
    h3.tick_params('x')
    h3.invert_yaxis()

    fig.tight_layout()
    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/P{0}R{1}_Incursions.pdf".format(n_poi, n_rovers))
    plt.close()


def create_skill_heatmap_plots(n_poi, n_rovers):
    # Plot Colors
    blue = np.array([26, 100, 255])/255
    orange = np.array([255, 130, 0])/255
    m_size = 0.001
    n_colors = int(1/m_size)
    hcolors = [orange, blue]
    hmap_colors = LinearSegmentedColormap.from_list("Custom", hcolors, N=n_colors)

    # File Paths
    r1_path = '../CKI/Output_Data/Rover0_SkillSelections.csv'
    r2_path = '../CKI/Output_Data/Rover1_SkillSelections.csv'
    r3_path = '../CKI/Output_Data/Rover2_SkillSelections.csv'
    r4_path = '../CKI/Output_Data/Rover3_SkillSelections.csv'
    r5_path = '../CKI/Output_Data/Rover4_SkillSelections.csv'
    r6_path = '../CKI/Output_Data/Rover5_SkillSelections.csv'

    rover_data = np.zeros((n_poi + 1, n_poi + 1))
    rov_input1 = []
    with open(r1_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input1.append(row)
    i = 0
    for row in rov_input1:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1
        i += 1

    rov_input2 = []
    with open(r2_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input2.append(row)
    i = 0
    for row in rov_input2:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1
        i += 1

    rov_input3 = []
    with open(r3_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input3.append(row)
    i = 0
    for row in rov_input3:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1
        i += 1

    rov_input4 = []
    with open(r4_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input4.append(row)
    i = 0
    for row in rov_input4:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1
        i += 1

    rov_input5 = []
    with open(r5_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input5.append(row)
    i = 0
    for row in rov_input5:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1
        i += 1

    rov_input6 = []
    with open(r6_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            rov_input6.append(row)
    i = 0
    for row in rov_input6:
        j = 0
        for val in row:
            rover_data[i, j] += float(val)
            j += 1
        i += 1

    for i in range(n_poi):
        for j in range(n_poi):
            if i == j:
                rover_data[n_poi, n_poi] += rover_data[i, j]
    rover_data[n_poi, n_poi] /= n_poi
    rover_data /= n_rovers

    # Create The Plot
    # x_axis = [0, 1, 2]
    # x_axis = [0, 1, 2, 3, 4]
    # x_axis = [0, 1, 2, 3, 4, 5]
    x_axis = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.imshow(rover_data, cmap=hmap_colors)
    plt.xticks(x_axis)
    plt.yticks(x_axis)
    plt.xlabel("Desired Policy Selection")
    plt.ylabel("Agent Policy Selection")
    plt.colorbar()

    # Add gridlines (requires shifting major/mior gridlines)
    ax = plt.gca()
    # Major ticks
    ax.set_xticks(np.arange(0, len(x_axis), 1))
    ax.set_yticks(np.arange(0, len(x_axis), 1))

    # Labels for major ticks
    # ax.set_xticklabels(['P1', 'P2', 'P0'])
    # ax.set_yticklabels(['P1', 'P2', 'P0'])
    # ax.set_xticklabels(['P1', 'P2', 'P3', 'P4', 'P0'])
    # ax.set_yticklabels(['P1', 'P2', 'P3', 'P4', 'P0'])
    # ax.set_xticklabels(['P1', 'P2', 'P3', 'P4', 'P5', 'P0'])
    # ax.set_yticklabels(['P1', 'P2', 'P3', 'P4', 'P5', 'P0'])
    ax.set_xticklabels(['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P0'])
    ax.set_yticklabels(['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P0'])

    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(x_axis), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(x_axis), 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)

    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/P{0}R{1}_Heatmap.pdf".format(n_poi, n_rovers))
    plt.close()


if __name__ == '__main__':
    graph_type = sys.argv[1]

    generations = 2000
    sample_rate = 20
    sruns = 30

    max_coupling = 6
    n_rovers = 6
    n_poi = 10

    if graph_type == "Coupling":
        create_coupling_plots(max_coupling, n_poi, n_rovers, sruns)
    elif graph_type == "Hazard":
        create_hazard_performance_plots(n_poi, n_rovers, sruns)
        create_hazard_incursion_plots(n_poi, n_rovers, sruns)
    elif graph_type == "HeatMap":
        create_skill_heatmap_plots(n_poi, n_rovers)
