//
// Created by zerbeln on 11/15/21.
//

#ifndef UNTITLED_MATRIX_MULTIPLY_H
#define UNTITLED_MATRIX_MULTIPLY_H

#include <iostream>
#include <cstdlib>
#include <vector>

using namespace std;

vector <double> nn_matrix_multiplication(vector <double> nn_layer, vector < vector <double> > weights, int m, int n){
    vector <double> output_vector;
    double sum;

    for (int i = 0; i < m; i++){
        sum = 0.0;
        for (int j = 0; j < n; j++){
            sum += nn_layer.at(j) * weights.at(i).at(j);
        }
        output_vector.push_back(sum);
    }

    return output_vector;
};

vector < vector <double> > reshape_vector(vector <double> in_vec, int m, int n){
    vector < vector <double> > matrix_vec;
    vector <double> temp;
    int count = 0;

    for (int i = 0; i < m; i++){
        temp.clear();
        for (int j = 0; j < n; j++){
            temp.push_back(in_vec.at(count));
            count++;
        }
        matrix_vec.push_back(temp);
    }

    return matrix_vec;
}

#endif //UNTITLED_MATRIX_MULTIPLY_H
