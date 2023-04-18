import csv
import numpy as np
import math
import pickle
import os


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


def get_standard_err_learning(csv_name, data_mean, generations, sample_rate, sruns):
    """
    Calculates standard deviation for training data. data_mean is an array of floats
    """
    csv_data = []
    with open(csv_name) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        for row in csv_reader:
            csv_data.append(row)

    n_data_points = int((generations/sample_rate) + 1)
    assert(len(data_mean) == n_data_points)

    temp_array = np.zeros(n_data_points)
    standard_dev = np.zeros(n_data_points)
    for row in csv_data:
        for i in range(n_data_points):
            temp_array[i] += (float(row[i]) - data_mean[i])**2

    temp_array /= sruns
    for i in range(n_data_points):
        standard_dev[i] = math.sqrt(temp_array[i])/math.sqrt(sruns)

    return standard_dev


def get_standard_err_performance(csv_name, data_mean, sruns):
    """
    Calculates standard deviation for performance data. data_mean is a float value (not an array)
    """

    csv_data = []
    with open(csv_name) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        for row in csv_reader:
            csv_data.append(row)

    temp_val = 0.0
    assert(len(csv_data[0]) == sruns)
    for row in csv_data:
        for val in row:
            temp_val += (float(val) - data_mean)**2

    standard_err = math.sqrt(temp_val/sruns)/math.sqrt(sruns)

    return standard_err


def get_acg_standard_err(csv_row, data_mean, sruns):
    temp_val = 0.0
    for val in csv_row:
        temp_val += (float(val) - data_mean)**2

    standard_err = math.sqrt(temp_val/sruns) / math.sqrt(sruns)

    return standard_err


def import_pickle_data(file_path):
    """
    Load saved Neural Network policies from pickle file
    """

    data_file = open(file_path, 'rb')
    pickle_data = pickle.load(data_file)
    data_file.close()

    return pickle_data
