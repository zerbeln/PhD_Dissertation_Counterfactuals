//
// Created by zerbeln on 1/12/22.
//

#ifndef C__CODE_LOCAL_REWARDS_H
#define C__CODE_LOCAL_REWARDS_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <iterator>
#include "parameters.h"
#include "agent.h"

using namespace std;


double toward_teammates_reward(int rover_id, vector <Rover> rovers){
    double reward = 0;
    double dist, rov_x, x, rov_y, y;

    rov_x = rovers.at(rover_id).rx;
    rov_y = rovers.at(rover_id).ry;

    for (int r = 0; r < n_rovers; r++){
        if (r != rovers.at(rover_id).self_id){
            x = rovers.at(r).rx;
            y = rovers.at(r).ry;

            dist = pow(rov_x - x, 2) + pow(rov_y - y, 2);

            reward -= dist;
        }
    }

    return reward;
}


double away_teammates_reward(int rover_id, vector <Rover> rovers){
    double reward = 0;
    double dist, rov_x, x, rov_y, y;

    rov_x = rovers.at(rover_id).rx;
    rov_y = rovers.at(rover_id).ry;

    for (int r = 0; r < n_rovers; r++){
        if (r != rovers.at(rover_id).self_id){
            x = rovers.at(r).rx;
            y = rovers.at(r).ry;

            dist = pow(rov_x - x, 2) + pow(rov_y - y, 2);

            reward += dist;
        }
    }

    return reward;
}


double toward_poi_reward(int rover_id, vector <Poi> pois){
    double reward = 0;
    double dist;

    for (int p = 0; p < n_poi; p++){
        dist = pois.at(p).observer_distances.at(rover_id);

        reward -= dist;
    }

    return reward;
}


double away_poi_reward(int rover_id, vector <Poi> pois){
    double reward = 0;
    double dist;

    for (int p = 0; p < n_poi; p++){
        dist = pois.at(p).observer_distances.at(rover_id);

        reward += dist;
    }

    return reward;
}


double two_poi_reward(int rover_id, vector <Poi> pois, int target_poi){
    double reward = 0;
    double dist;

    dist = pois.at(target_poi).observer_distances.at(rover_id);
    if (dist < obs_radius){
        reward += pois.at(target_poi).poi_value / dist;
    }

    return reward;
}


double four_quadrant_rewards(int rover_id, vector <Poi> pois, int target_quadrant){
    double reward = 0;
    double dist;

    for (int p = 0; p < n_poi; p++){
        if (target_quadrant == pois.at(p).quadrant){
            dist = pois.at(p).observer_distances.at(rover_id);

            if (dist < obs_radius){
                reward += pois.at(p).poi_value/dist;
            }
        }
    }

    return reward;
}

#endif //C__CODE_LOCAL_REWARDS_H
