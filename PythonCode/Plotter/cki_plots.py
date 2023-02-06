import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import csv
import sys
import os
from plots_common_functions import import_reward_data, get_standard_deviations


def calc_max_poi_val(n_rovers, poi_coupling):
    poi_path = '../C{0}/CBA/World_Config/POI_Config.csv'.format(poi_coupling)

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
    cki_data = []

    for i in range(max_coupling):
        g_path = '../C{0}/Global/Output_Data/Final_GlobalRewards.csv'.format(i+1)
        d_path = '../C{0}/Difference/Output_Data/Final_GlobalRewards.csv'.format(i+1)
        dpp_path = '../C{0}/D++/Output_Data/Final_GlobalRewards.csv'.format(i+1)
        cki_path = '../C{0}/CBA/Output_Data/Final_GlobalRewards.csv'.format(i+1)

        max_reward = calc_max_poi_val(n_rovers, i+1)

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
                temp.append(float(val)/max_reward)
        g_data.append(np.mean(temp))

        # Difference Reward Performance --------------------------
        with open(d_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                d_input.append(row)
        temp = []
        for row in d_input:
            for val in row:
                temp.append(float(val)/max_reward)
        d_data.append(np.mean(temp))

        # D++ Reward Performance ------------------------------
        with open(dpp_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                dpp_input.append(row)
        temp = []
        for row in dpp_input:
            for val in row:
                temp.append(float(val)/max_reward)
        dpp_data.append(np.mean(temp))

        # CKI Performance -------------------------------------
        with open(cki_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                cki_input.append(row)
        temp = []
        for row in cki_input:
            for val in row:
                temp.append(float(val)/max_reward)
        cki_data.append(np.mean(temp))

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

    plt.plot(x_axis, g_data, color=color1, linestyle='--', marker='^')
    plt.plot(x_axis, d_data, color=color2, linestyle='-.', marker='o')
    plt.plot(x_axis, dpp_data, color=color3, linestyle=':', marker='s')
    plt.plot(x_axis, cki_data, color=color4, marker='H')

    plt.xlabel("POI Coupling Requirement")
    plt.ylabel("Percentage of Maximum Score")
    plt.legend(["G", "D", "D++", "CKI"])

    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/P{0}R{1}_Coupling.pdf".format(n_poi, n_rovers))
    plt.close()


def create_hazard_performance_plots(n_poi, n_rovers):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia
    colors = [color3, color2, color4, color1, color5]
    x_axis = ["G", "D", "CKI-C1", "D++", "CKI-C3"]

    g_data = []
    d_data = []
    dpp_data = []
    cki_c1_data = []
    cki_c3_data = []

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

    h1_ydata = [g_data[0], d_data[0], cki_c1_data[0], dpp_data[0], cki_c3_data[0]]
    h2_ydata = [g_data[1], d_data[1], cki_c1_data[1], dpp_data[1], cki_c3_data[1]]
    h3_ydata = [g_data[2], d_data[2], cki_c1_data[2], dpp_data[2], cki_c3_data[2]]

    fig, (h1, h2, h3) = plt.subplots(nrows=1, ncols=3)

    h1.bar(x_axis, h1_ydata, color=colors)
    h1.axhline(0, color='black')
    h1.axvline(2.5, color='k', linestyle="--")
    h1.set_ylabel("Average Team Reward")
    h1.set_title("3 Hazards")
    h1.tick_params('x', labelrotation=45)

    h2.bar(x_axis, h2_ydata, color=colors)
    h2.axhline(0, color='black')
    h2.axvline(2.5, color='k', linestyle="--")
    h2.set_title("4 Hazards")
    h2.tick_params('x', labelrotation=45)

    h3.bar(x_axis, h3_ydata, color=colors)
    h3.axhline(0, color='black')
    h3.axvline(2.5, color='k', linestyle="--")
    h3.set_title("6 Hazards")
    h3.tick_params('x', labelrotation=45)

    fig.tight_layout()
    # Save the plot
    if not os.path.exists('Plots'):  # If Data directory does not exist, create it
        os.makedirs('Plots')
    plt.savefig("Plots/P{0}R{1}_Hazards.pdf".format(n_poi, n_rovers))
    plt.close()


def create_hazard_incursion_plots(n_poi, n_rovers):
    # Plot Color Palette
    color1 = np.array([26, 133, 255]) / 255  # Blue
    color2 = np.array([255, 194, 10]) / 255  # Yellow
    color3 = np.array([230, 97, 0]) / 255  # Orange
    color4 = np.array([93, 58, 155]) / 255  # Purple
    color5 = np.array([211, 95, 183]) / 255  # Fuschia
    colors = [color3, color2, color4, color1, color5]
    x_axis = ["G", "D", "CKI-C1", "D++", "CKI-C3"]

    g_data = []
    d_data = []
    dpp_data = []
    cki_c1_data = []
    cki_c3_data = []

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

    h1_ydata = [g_data[0], d_data[0], cki_c1_data[0], dpp_data[0], cki_c3_data[0]]
    h2_ydata = [g_data[1], d_data[1], cki_c1_data[1], dpp_data[1], cki_c3_data[1]]
    h3_ydata = [g_data[2], d_data[2], cki_c1_data[2], dpp_data[2], cki_c3_data[2]]

    fig, (h1, h2, h3) = plt.subplots(nrows=3, ncols=1)

    h1.barh(x_axis, h1_ydata, color=colors)
    h1.axhline(2.5, color='k', linestyle='--')
    h1.set_title("3 Hazards")
    h1.tick_params('x')
    h1.invert_yaxis()

    h2.barh(x_axis, h2_ydata, color=colors)
    h2.axhline(2.5, color='k', linestyle='--')
    h2.set_title("4 Hazards")
    h2.tick_params('x')
    h2.invert_yaxis()

    h3.barh(x_axis, h3_ydata, color=colors)
    h3.axhline(2.5, color='k', linestyle='--')
    h3.set_title("6 Hazards")
    h3.set_xlabel("Number of Rover Incursions")
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
    r1_path = '../CBA/Output_Data/Rover0_SkillSelections.csv'
    r2_path = '../CBA/Output_Data/Rover1_SkillSelections.csv'
    r3_path = '../CBA/Output_Data/Rover2_SkillSelections.csv'
    r4_path = '../CBA/Output_Data/Rover3_SkillSelections.csv'
    r5_path = '../CBA/Output_Data/Rover4_SkillSelections.csv'
    r6_path = '../CBA/Output_Data/Rover5_SkillSelections.csv'

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


    rover_data /= n_rovers

    # Create The Plot
    x_axis = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.imshow(rover_data, cmap=hmap_colors)
    plt.xticks(x_axis, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '0'])
    plt.yticks(x_axis, ['POI 1', 'POI 2', 'POI 3', 'POI 4', 'POI 5', 'POI 6', 'POI 7', 'POI 8', 'POI 9', 'POI 10', 'Stop'])
    plt.xlabel("Counterfactual States")
    plt.ylabel("Agent Skill")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    graph_type = sys.argv[1]

    generations = 2000
    sample_rate = 20
    sruns = 30

    max_coupling = 6
    n_rovers = 6
    n_poi = 10

    if graph_type == "Coupling":
        create_coupling_plots(max_coupling, n_poi, n_rovers)
    elif graph_type == "Hazard":
        create_hazard_performance_plots(n_poi, n_rovers)
        create_hazard_incursion_plots(n_poi, n_rovers)
    elif graph_type == "HeatMap":
        create_skill_heatmap_plots(n_poi, n_rovers)
