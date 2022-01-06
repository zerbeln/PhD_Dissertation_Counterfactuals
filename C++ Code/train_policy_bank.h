//
// Created by zerbeln on 12/6/21.
//

#ifndef UNTITLED_TRAIN_POLICY_BANK_H
#define UNTITLED_TRAIN_POLICY_BANK_H

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include "rover_domain.h"
#include "ccea.h"
#include "parameters.h"

void record_reward_history(vector <double> reward_vec){
    ofstream csvfile;

    int vec_size = reward_vec.size();

    csvfile.open("Output_Data/RewardHistory.csv", fstream::app);
    for (int i = 0; i < vec_size; i++){
        csvfile << reward_vec.at(i) << ",";
    }
    csvfile << "\n";
    csvfile.close();

}

void train_four_quadrants(){

}

#endif //UNTITLED_TRAIN_POLICY_BANK_H
