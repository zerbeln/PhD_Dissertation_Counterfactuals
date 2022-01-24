//
// Created by zerbeln on 11/15/21.
//

#ifndef UNTITLED_CCEA_H
#define UNTITLED_CCEA_H

#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>
#include <algorithm>
#include "parameters.h"

using namespace std;


class Ccea{
    int n_elites = num_elites;
    int n_weights = ((n_inputs+1) * n_hidden) + ((n_hidden+1) * n_outputs);
    double epsilon = eps;
    double mut_chance = mutation_chance;
    double mut_rate = mutation_rate;
public:
    vector < vector <double> > population;
    vector <double> fitness;
    vector <int> team_selection;


    void create_new_population(){
        population.clear();
        fitness.clear();

        cauchy_distribution <double> dist(0.0, 1.0);
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        vector <double> temp;

        for (int p = 0; p < pop_size; p++){

            fitness.push_back(0.0);
            temp.clear();
            for (int w = 0; w < n_weights; w++){
                temp.push_back(dist(generator));
            }
            population.push_back(temp);
        }
    }

    void reset_fitness(){
        for (int p = 0; p < pop_size; p++){
            fitness.at(p) = 0.0;
        }
    }

    void mutate_weights(){
        int pol_id = n_elites;
        double weight;
        float rnum;

        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        normal_distribution <double> dist(0.0, mut_rate);

        while (pol_id < pop_size){
            for (int w = 0; w < n_weights; w++){
                rnum = float(rand()/(RAND_MAX + 1.0));
                if (rnum < mut_chance){
                    weight = population.at(pol_id).at(w);
                    weight += dist(generator) * weight;
                    population.at(pol_id).at(w) = weight;
                }
            }

            pol_id++;
        }
    }

    void epsilon_greedy_selection(){
        vector < vector <double> > new_pop;
        int parent;
        double rnum;

        for (int pol_id = 0; pol_id < pop_size; pol_id++){
            if (pol_id < n_elites){
                new_pop.push_back(population.at(pol_id));
            }
            else{
                rnum = float(rand()/(RAND_MAX + 1.0));
                if (rnum < epsilon){
                    new_pop.push_back(population.at(0));
                }
                else{
                    parent = rand() % pop_size;
                    new_pop.push_back(population.at(parent));
                }
            }
        }

        population = new_pop;
    }

    void select_policy_teams(){
        int team_id, t;
        team_selection.clear();

        t = 0;
        while (t < pop_size){
            team_id = rand() % pop_size;
            if (find(team_selection.begin(), team_selection.end(), team_id) == team_selection.end()){
                team_selection.push_back(team_id);
                t++;
            }
        }
    }

    void rank_population(){
        vector < vector <double> > ranked_pop;
        int pol_a, pol_b;

        ranked_pop = population;
        for (pol_a = 0; pol_a < pop_size; pol_a++){
            pol_b = pol_a+1;
            ranked_pop.at(pol_a) = population.at(pol_a);
            while (pol_b < pop_size){
                if (pol_a != pol_b){
                    if (fitness.at(pol_a) < fitness.at(pol_b)){
                        fitness.at(pol_a), fitness.at(pol_b) = fitness.at(pol_b), fitness.at(pol_a);
                        ranked_pop.at(pol_a) = population.at(pol_b);
                    }
                }
                pol_b++;
            }
        }

        population = ranked_pop;
    }

    void down_select(){
        rank_population();
        epsilon_greedy_selection();
        mutate_weights();
    }

};

#endif //UNTITLED_CCEA_H
