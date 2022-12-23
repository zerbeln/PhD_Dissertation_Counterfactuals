import csv
import numpy as np
import math


def import_reward_data(csv_name, generations, sample_rate, sruns):

    config_input = []
    with open(csv_name) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        for row in csv_reader:
            config_input.append(row)

    n_data_points = int((generations/sample_rate) + 1)
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
