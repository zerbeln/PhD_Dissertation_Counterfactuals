//
// Created by zerbeln on 11/8/21.
//

#ifndef UNTITLED_ROVER_DOMAIN_H
#define UNTITLED_ROVER_DOMAIN_H

#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include "agent.h"
#include "parameters.h"

using namespace std;

class RoverDomain {
    friend class Poi;
    friend class Rover;
public:
    vector <Poi> pois;
    vector <Rover> rovers;

    void load_world(){
        string line;

        ifstream csvfile("World_Config/POI_Config.csv", ios::in);

        if(!csvfile){
            cout << "ERROR: CSV FILE NOT OPEN" << endl;
        }

        for(int i = 0; i < n_poi; i++){
            Poi p;
            pois.push_back(p);
            getline(csvfile, line, ',');
            pois.at(i).x_position = stod(line);
            getline(csvfile, line, ',');
            pois.at(i).y_position =  stod(line);
            getline(csvfile, line, ',');
            pois.at(i).poi_value = stod(line);
            getline(csvfile, line, ',');
            pois.at(i).coupling = stoi(line);
            getline(csvfile, line, '\n');
            pois.at(i).quadrant = stoi(line);

            for (int r = 0; r < n_rovers; r++){
                pois.at(i).observer_distances.push_back(0.0);
            }
        }

        csvfile.close();
    }

    void load_rover() {
        string line;
        ifstream csvfile("World_Config/Rover_Config.csv", ios::in);

        if (!csvfile) {
            cout << "ERROR: CSV FILE NOT OPEN" << endl;
        }

        for(int i = 0; i < n_rovers; i++){
            Rover r;
            rovers.push_back(r);
            rovers.at(i).self_id = i;
            getline(csvfile, line, ',');
            rovers.at(i).init_pos.push_back(stod(line));
            getline(csvfile, line, ',');
            rovers.at(i).init_pos.push_back(stod(line));
            getline(csvfile, line, '\n');
            rovers.at(i).init_pos.push_back(stod(line));
        }
        csvfile.close();
    }

    double calc_global(){
        double global_reward = 0.0;
        double poi_coupling, summed_dist;
        int observer_count;

        vector <double> rover_distances;
        vector <double> observers;

        for (int p = 0; p < n_poi; p++){
            poi_coupling = double(pois.at(p).coupling);
            observer_count = 0;
            rover_distances = pois.at(p).observer_distances;
            sort(rover_distances.begin(), rover_distances.end());

            observers.clear();
            for(int c = 0; c < pois.at(p).coupling; c++){
                if (rover_distances.at(c) <= obs_radius){
                    observers.push_back(rover_distances.at(c));
                    observer_count++;
                }
            }

            summed_dist = 0.0;
            if (observer_count >= pois.at(p).coupling){
                summed_dist = accumulate(observers.begin(), observers.end(), 0.0);
                global_reward += pois.at(p).poi_value / (summed_dist/poi_coupling);
            }
        }

        return global_reward;
    }
};

#endif //UNTITLED_ROVER_DOMAIN_H
