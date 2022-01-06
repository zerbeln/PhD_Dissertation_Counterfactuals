//
// Created by zerbeln on 11/16/21.
//

#ifndef UNTITLED_PARAMETERS_H
#define UNTITLED_PARAMETERS_H

#include <iostream>

using namespace std;

// Test Parameters
int generations = 5000;
int stat_runs = 15;
int sample_rate = 20;

// Domain Parameters
double world_x = 100.0;
double world_y = 100.0;
int n_poi = 10;
int n_rovers = 6;
double obs_radius = 4.0;
int coupling = 1;
double delta_min = 1.0;
int pop_size = 30;
int rover_steps = 25;

// Neural Network Parameters
int n_inputs = 8;
int n_hidden = 10;
int n_outputs = 2;


#endif //UNTITLED_PARAMETERS_H
