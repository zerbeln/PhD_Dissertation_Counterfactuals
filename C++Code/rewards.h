//
// Created by zerbeln on 11/16/21.
//

#ifndef UNTITLED_REWARDS_H
#define UNTITLED_REWARDS_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <iterator>
#include "parameters.h"
#include "agent.h"

using namespace std;

vector <double> calc_difference(vector <Poi> pois, double g_reward){
    vector <double> difference_rewards;
    vector <double> rover_distances;
    vector <double> observers;

    double counterfactual_g_reward, poi_coupling, dist, summed_dist;
    int observer_count;

    for (int a = 0; a < n_rovers; a++){
        counterfactual_g_reward = 0.0;

        for (int p = 0; p < n_poi; p++){
            poi_coupling = double(pois.at(p).coupling);
            observer_count = 0;
            rover_distances = pois.at(p).observer_distances;
            rover_distances.at(a) = 1000.00;
            sort(rover_distances.begin(), rover_distances.end());

            observers.clear();
            for (int i = 0; i < pois.at(p).coupling; i++){
                dist = rover_distances.at(i);
                if(dist < obs_radius){
                    observers.push_back(dist);
                    observer_count++;
                }
            }

            summed_dist = 0.0;
            if (observer_count >= pois.at(p).coupling){
                summed_dist = accumulate(observers.begin(), observers.end(), 0.0);
                counterfactual_g_reward = pois.at(p).poi_value / (summed_dist/poi_coupling);
            }
        }

        difference_rewards.push_back(g_reward - counterfactual_g_reward);
    }

    return difference_rewards;
}


vector <double> calc_dpp(vector <Poi> pois, double g_reward){
    vector <double> dpp_rewards;
    vector <double> d_rewards;
    vector <double> rover_distances;
    vector <double> observers;

    int observer_count, n_counters;
    double temp_dpp, counterfactual_g_reward, dist, summed_dist, n_count, poi_coupling;

    d_rewards = calc_difference(pois, g_reward);
    n_counters = coupling -1 ;
    n_count = double(coupling-1);
    for (int a = 0; a < n_rovers; a++){
        counterfactual_g_reward = 0.0;
        for(int p = 0; p < n_poi; p++){
            observer_count = 0;
            poi_coupling = double(pois.at(p).coupling);
            rover_distances = pois.at(p).observer_distances;
            for (int n = 0; n < n_counters; n++){
                rover_distances.push_back(rover_distances.at(a));
            }
            sort(rover_distances.begin(), rover_distances.end());

            observers.clear();
            for (int i = 0; i < pois.at(p).coupling; i++){
                dist = rover_distances.at(i);
                if (rover_distances.at(i) < obs_radius){
                    observers.push_back(dist);
                    observer_count++;
                }
            }

            summed_dist = 0.0;
            if (observer_count >= pois.at(p).coupling){
                summed_dist = accumulate(observers.begin(), observers.end(), 0.0);
                counterfactual_g_reward = pois.at(p).poi_value / (summed_dist/poi_coupling);
            }
        }
        dpp_rewards.push_back((counterfactual_g_reward - g_reward)/n_count);
    }

    for (int a = 0; a < n_rovers; a++){
        if (dpp_rewards.at(a) > d_rewards.at(a)){
            for(int c = 0; c < coupling; c++){
                n_counters = c + 1;
                n_count = double(n_counters);
                counterfactual_g_reward = 0.0;

                for (int p = 0; p < n_poi; p++){
                    observer_count = 0;
                    poi_coupling = double(pois.at(p).coupling);
                    rover_distances = pois.at(p).observer_distances;
                    for (int n = 0; n < n_counters; n++){
                        rover_distances.push_back(rover_distances.at(a));
                    }
                    sort(rover_distances.begin(), rover_distances.end());

                    observers.clear();
                    for (int i = 0; i < pois.at(p).coupling; i++){
                        dist = rover_distances.at(i);
                        if (rover_distances.at(i) < obs_radius){
                            observers.push_back(dist);
                            observer_count++;
                        }
                    }

                    summed_dist = 0.0;
                    if (observer_count >= pois.at(p).coupling){
                        summed_dist = accumulate(observers.begin(), observers.end(), 0.0);
                        counterfactual_g_reward = pois.at(p).poi_value / (summed_dist/poi_coupling);
                    }
                }
                temp_dpp = (counterfactual_g_reward - g_reward)/n_count;
                if (temp_dpp > dpp_rewards.at(a)){
                    dpp_rewards.at(a) = temp_dpp;
                    c = coupling + 1;
                }
            }
        }
        else{
            dpp_rewards.at(a) = d_rewards.at(a);
        }
    }

    return dpp_rewards;
}

#endif //UNTITLED_REWARDS_H
