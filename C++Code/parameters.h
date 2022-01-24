//
// Created by zerbeln on 11/16/21.
//

#ifndef UNTITLED_PARAMETERS_H
#define UNTITLED_PARAMETERS_H

#include <iostream>

using namespace std;

// Test Parameters
int stat_runs = 50;
int generations = 3000;
int pbank_generations = 10;
string reward_type = "Global";  // Global, Difference, or DPP
int coupling = 1;
int sample_rate = 20;

//Domain Parameters
double world_x = 100.0;
double world_y = 100.0;
int n_poi = 10;
int n_rovers = 6;
int rover_steps = 25;

// Rover Parameters
string sensor_model = "summed";
double obs_radius = 4.0;
double delta_min = 1.0;
double delta_max = 2.5;
double angle_res = 90.0;

// Suggestion Parameters
string policy_bank_type = "Two_Poi";
int n_policies = 2;
int n_suggestions = 2;
int s_inputs = 16;
int s_hidden = 12;
int s_outputs = n_policies;

// Neural Network Parameters
int n_inputs = 8;
int n_hidden = 10;
int n_outputs = 2;

// CCEA Parameters
int pop_size = 30;
double mutation_chance = 0.1;
double mutation_rate = 0.1;
double eps = 0.1;
int num_elites = 1;

#endif //UNTITLED_PARAMETERS_H
