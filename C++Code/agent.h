//
// Created by zerbeln on 11/5/21.
//

#ifndef UNTITLED_AGENT_H
#define UNTITLED_AGENT_H

#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <iterator>
#include "matrix_multiply.h"
#include "parameters.h"
#include "calc_angle_dist.h"

using namespace std;

class Poi {
public:
    double x_position;
    double y_position;
    double poi_value;
    int coupling;
    int quadrant;

    vector <double> observer_distances;
};

class Rover {
    friend class Poi;
    string sensor_type = sensor_model;
    double sensor_range = obs_radius;
    double sensor_res = angle_res;
    double dmax = delta_max;
    int n_weights = ((n_inputs+1) * n_hidden) + ((n_hidden+1) * n_outputs);
    int n_layer1 = ((n_inputs+1) * n_hidden);
    int n_cbm_weights = ((s_inputs + 1) * s_hidden) + ((s_hidden + 1) * s_outputs);
    int n_cbm_layer1 = ((s_inputs + 1) + s_hidden);

public:
    int self_id;
    double rx;
    double ry;
    double rtheta;
    vector <double> init_pos;
    vector <double> poi_distances;
    vector <double> sensor_readings;

    //ROVER NEURAL NETWORK
    vector <double> input_layer;
    vector <double> hidden_layer;
    vector <double> output_layer;
    vector < vector <double> > weights_l1;
    vector < vector <double> > weights_l2;

    //ROVER CBM NETWORK
    vector <double> cbm_input_layer;
    vector <double> cbm_hidden_layer;
    vector <double> cbm_output_layer;
    vector < vector <double> > cbm_weights_l1;
    vector < vector <double> > cbm_weights_l2;
    vector < vector <double> > policy_bank;

    // Rover Functions
    void reset_rover() {
        rx = init_pos.at(0);
        ry = init_pos.at(1);
        rtheta = init_pos.at(2);
    }

    void rover_step() {
        double dx, dy, x, y;

        dx = 2 * dmax * (output_layer.at(0) - 0.5);
        dy = 2 * dmax * (output_layer.at(1) - 0.5);

        // Update X Position
        x = dx + rx;
        if (x < 0.0) {
            x = 0.0;
        } else if (x > world_x - 1.0) {
            x = world_x - 1.0;
        }

        // Update Y Position
        y = dy + ry;
        if (y < 0.0) {
            y = 0.0;
        } else if (y > world_y - 1.0) {
            y = world_y - 1.0;
        }

        rx = x;
        ry = y;
    }

    vector <double> poi_scan(vector <Poi> pois) {
        vector <double> poi_state(4, 0.0);
        vector < vector <double> > temp_poi_dist_list(4);
        double angle, dist;
        int bracket, num_poi_bracket;
        poi_distances.clear();

        for (int p = 0; p < n_poi; p++) {
            angle, dist = get_angle_dist(rx, ry, pois.at(p).x_position, pois.at(p).y_position);
            poi_distances.push_back(sqrt(dist));
            bracket = int(angle / sensor_res);
            if (bracket > 3) {
                bracket -= 4;
            }

            temp_poi_dist_list.at(bracket).push_back(pois.at(p).poi_value / dist);
        }

        for (bracket = 0; bracket < 4; bracket++) {
            num_poi_bracket = temp_poi_dist_list.at(bracket).size();
            if (num_poi_bracket > 0) {
                if (sensor_type == "summed"){
                    poi_state.at(bracket) = double(accumulate(temp_poi_dist_list.at(bracket).begin(), temp_poi_dist_list.at(bracket).end(), 0.0));
                }
                else if (sensor_type == "density"){
                    poi_state.at(bracket) = double(accumulate(temp_poi_dist_list.at(bracket).begin(), temp_poi_dist_list.at(bracket).end(), 0.0) / num_poi_bracket);
                }
            } else {
                poi_state.at(bracket) = -1.0;
            }
        }

        return poi_state;
    }

    vector<double> rover_scan(vector <Rover> rovers) {
        vector <double> rover_state(4, 0.0);
        vector <vector <double> > temp_rover_dist_list(4);
        double angle, dist, tx, ty;
        int bracket, num_rover_bracket;

        for (int r = 0; r < n_rovers; r++) {
            if (r != self_id){
                tx = rovers.at(r).rx;
                ty = rovers.at(r).ry;
                angle, dist = get_angle_dist(rx, ry, tx, ty);
                bracket = int(angle / sensor_res);
                if (bracket > 3) {
                    bracket -= 4;
                }

                temp_rover_dist_list.at(bracket).push_back(1 / dist);
            }

        }

        for (bracket = 0; bracket < 4; bracket++) {
            num_rover_bracket = temp_rover_dist_list.at(bracket).size();
            if (num_rover_bracket > 0) {
                if (sensor_type == "summed"){
                    rover_state.at(bracket) = double(accumulate(temp_rover_dist_list.at(bracket).begin(), temp_rover_dist_list.at(bracket).end(), 0.0));
                }
                else if (sensor_type == "density"){
                    rover_state.at(bracket) = double(accumulate(temp_rover_dist_list.at(bracket).begin(), temp_rover_dist_list.at(bracket).end(), 0.0) / num_rover_bracket);
                }
            } else {
                rover_state.at(bracket) = -1.0;
            }
        }

        return rover_state;
    }

    void scan_environment(vector <Poi> pois, vector <Rover> rovers) {
        vector <double> poi_state = poi_scan(pois);
        vector <double> rover_state = rover_scan(rovers);

        sensor_readings.clear();
        for (int i = 0; i < 4; i++) {
            sensor_readings.push_back(poi_state.at(i));
            sensor_readings.push_back(rover_state.at(i));
        }
    }

    // Motor Control NN ---------------------------------------------------------------------------------
    double sigmoid(double inp) {
        double sig;

        sig = 1 / (1 + exp(-inp));

        return sig;
    }

    void get_weights(vector <double> nn_weights) {
        weights_l1.clear();
        weights_l2.clear();
        vector <double> temp1;
        vector <double> temp2;

        // Layer 1 Weights
        for(int i = 0; i < n_layer1; i++){
            temp1.push_back(nn_weights.at(i));
        }
        weights_l1 = reshape_vector(temp1, n_hidden, n_inputs+1);

        // Layer 2 Weights
        for(int i = n_layer1; i < n_weights; i++){
            temp2.push_back(nn_weights.at(i));
        }
        weights_l2 = reshape_vector(temp2, n_outputs, n_hidden+1);
    }

    void clear_layers(){
        input_layer.clear();
        hidden_layer.clear();
        output_layer.clear();
    }

    void get_nn_outputs() {
        clear_layers();

        for (int i = 0; i < n_inputs; i++){
            input_layer.push_back(sensor_readings.at(i));
        }
        input_layer.push_back(1.0);

        hidden_layer = nn_matrix_multiplication(input_layer, weights_l1, n_hidden, n_inputs+1);
        hidden_layer.push_back(1.0);  // Biasing node
        for (int i = 0; i < n_hidden+1; i++) {
            hidden_layer.at(i) = sigmoid(hidden_layer.at(i));
        }

        output_layer = nn_matrix_multiplication(hidden_layer, weights_l2, n_outputs, n_hidden+1);
        for (int i = 0; i < n_outputs; i++) {
            output_layer.at(i) = sigmoid(output_layer.at(i));
        }
    }

    //ROVER CBM NETWORK ----------------------------------------------------------------
    void cbm_get_weights(vector <double> nn_weights) {
        cbm_weights_l1.clear();
        cbm_weights_l2.clear();
        vector <double> temp1;
        vector <double> temp2;

        // Layer 1 Weights
        for(int i = 0; i < n_cbm_layer1; i++){
            temp1.push_back(nn_weights.at(i));
        }
        cbm_weights_l1 = reshape_vector(temp1, n_hidden, n_inputs+1);

        // Layer 2 Weights
        for(int i = n_cbm_layer1; i < n_cbm_weights; i++){
            temp2.push_back(nn_weights.at(i));
        }
        cbm_weights_l2 = reshape_vector(temp2, n_outputs, n_hidden+1);
    }

    void cbm_clear_layers(){
        cbm_input_layer.clear();
        cbm_hidden_layer.clear();
        cbm_output_layer.clear();
    }

    void cbm_get_nn_outputs(vector <double> cbm_state) {
        clear_layers();

        for (int i = 0; i < s_inputs; i++){
            cbm_input_layer.push_back(cbm_state.at(i));
        }
        cbm_input_layer.push_back(1.0);

        cbm_hidden_layer = nn_matrix_multiplication(cbm_input_layer, cbm_weights_l1, s_hidden, s_inputs+1);
        cbm_hidden_layer.push_back(1.0);  // Biasing node
        for (int i = 0; i < s_hidden+1; i++) {
            cbm_hidden_layer.at(i) = sigmoid(cbm_hidden_layer.at(i));
        }

        cbm_output_layer = nn_matrix_multiplication(cbm_hidden_layer, cbm_weights_l2, s_outputs, s_hidden+1);
        for (int i = 0; i < n_outputs; i++) {
            cbm_output_layer.at(i) = sigmoid(cbm_output_layer.at(i));
        }
    }

};

#endif //UNTITLED_AGENT_H
