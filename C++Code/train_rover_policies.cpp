//
// Created by zerbeln on 1/11/22.
//

#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include <random>
#include "rover_domain.h"
#include "ccea.h"
#include "parameters.h"
#include "local_rewards.h"

using namespace std;

bool IsPathExist(const string &s){
    struct stat buffer;
    return (stat (s.c_str(), &buffer) == 0);
}

void check_directories(const char* path_name){
    int isdir;

    // Check if directory exists
    isdir = IsPathExist(path_name);

    // If not, create directory
    if (isdir == 0){
        mkdir(path_name, 0777);
    }
}

void record_reward_history(vector <double> reward_vec, int srun){
    ofstream csvfile;
    string file_path = "Output_Data/RewardHistory.csv";
    int vec_size = reward_vec.size();

    // Check for directory on first stat_run
    if (srun == 0){
        check_directories("Output_Data");
    }

    csvfile.open(file_path, fstream::app);
    for (int i = 0; i < vec_size; i++){
        csvfile << reward_vec.at(i) << ",";
    }
    csvfile << "\n";
    csvfile.close();

}

string int_to_str(int x) {
    stringstream ss;
    ss << x;
    return ss.str();
}

void save_best_policies(int rover_id, int srun, vector <double> policy, string filename){
    ofstream txtfile;

    string dir_path = "Policy_Bank/Rover" + int_to_str(rover_id);
    string filepath = dir_path + "/" + filename + int_to_str(srun) + ".txt";

    // Check if directories exist on first stat run
    if (srun == 0){
        check_directories("Policy_Bank");
        check_directories(dir_path.data());
    }

    txtfile.open(filepath);
    int vec_size = policy.size();

    for (int i = 0; i < vec_size; i++){
        txtfile << policy.at(i) << ",";
    }
    txtfile.close();
}

void train_toward_pois() {
    double reward;
    int policy_id;
    vector<double> nn_weights;
    vector<vector<double> > rover_rewards;
    vector<double> local_rewards;
    string fname = "TowardPoisPolicy";

    RoverDomain rd;
    rd.load_world();
    rd.load_rover();

    // Initialize CCEA population for each rover
    vector<Ccea> pops;
    for (int r = 0; r < n_rovers; r++) {
        Ccea ea;
        pops.push_back(ea);
    }

    cout << "Training Go Towards POIs Policy" << endl;
    for (int s = 0; s < stat_runs; s++) {
        cout << "Stat Run: " << s << endl;
        for (int r = 0; r < n_rovers; r++) {
            pops.at(r).create_new_population();
            rd.rovers.at(r).reset_rover();
        }

        for (int gen = 0; gen < generations; gen++) {
            for (int r = 0; r < n_rovers; r++) {
                pops.at(r).select_policy_teams();
            }

            // Test randomly selected teams
            for (int t = 0; t < pop_size; t++) {
                for (int r = 0; r < n_rovers; r++) {
                    rd.rovers.at(r).reset_rover();
                }

                // Get rover control weights and conduct initial scan
                for (int r = 0; r < n_rovers; r++) {
                    policy_id = pops.at(r).team_selection.at(t);
                    nn_weights = pops.at(r).population.at(policy_id);
                    rd.rovers.at(r).get_weights(nn_weights);
                    rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                    rd.rovers.at(r).get_nn_outputs();
                }

                // Rovers move for number of time steps
                rover_rewards.clear();
                for (int stp = 0; stp < rover_steps; stp++) {
                    local_rewards.clear();
                    for (int r = 0; r < n_rovers; r++) {
                        rd.rovers.at(r).rover_step();
                    }
                    for (int r = 0; r < n_rovers; r++) {
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        for (int p = 0; p < n_poi; p++) {
                            rd.pois.at(p).observer_distances.at(r) = rd.rovers.at(r).poi_distances.at(p);
                        }
                        rd.rovers.at(r).get_nn_outputs();
                        reward = toward_poi_reward(r, rd.pois);
                        local_rewards.push_back(reward);
                    }
                    rover_rewards.push_back(local_rewards);

                }

                for (int r = 0; r < n_rovers; r++) {
                    policy_id = pops.at(r).team_selection.at(t);
                    reward = accumulate(rover_rewards.at(r).begin(), rover_rewards.at(r).end(), 0.0);
                    pops.at(r).fitness.at(policy_id) = reward;
                }
            }

            for (int r = 0; r < n_rovers; r++) {
                pops.at(r).down_select();
            }
        }

        for (int rov_id = 0; rov_id < n_rovers; rov_id++) {
            nn_weights = pops.at(rov_id).population.at(0);
            save_best_policies(rov_id, s, nn_weights, fname);
        }
    }
}

void train_away_pois(){
    double reward;
    int policy_id;
    vector <double> nn_weights;
    vector < vector <double> > rover_rewards;
    vector <double> local_rewards;
    string fname = "AwayPoisPolicy";

    RoverDomain rd;
    rd.load_world();
    rd.load_rover();

    // Initialize CCEA population for each rover
    vector <Ccea> pops;
    for (int r = 0; r < n_rovers; r++){
        Ccea ea;
        pops.push_back(ea);
    }

    cout << "Training Go Away From POIs Policy" << endl;
    for (int s = 0; s < stat_runs; s++) {
        cout << "Stat Run: " << s << endl;
        for (int r = 0; r < n_rovers; r++) {
            pops.at(r).create_new_population();
            rd.rovers.at(r).reset_rover();
        }

        for (int gen = 0; gen < generations; gen++) {
            for (int r = 0; r < n_rovers; r++) {
                pops.at(r).select_policy_teams();
            }

            // Test randomly selected teams
            for (int t = 0; t < pop_size; t++) {
                for (int r = 0; r < n_rovers; r++) {
                    rd.rovers.at(r).reset_rover();
                }

                // Get rover control weights and conduct initial scan
                for (int r = 0; r < n_rovers; r++) {
                    policy_id = pops.at(r).team_selection.at(t);
                    nn_weights = pops.at(r).population.at(policy_id);
                    rd.rovers.at(r).get_weights(nn_weights);
                    rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                    rd.rovers.at(r).get_nn_outputs();
                }

                // Rovers move for number of time steps
                rover_rewards.clear();
                for (int stp = 0; stp < rover_steps; stp++) {
                    local_rewards.clear();
                    for (int r = 0; r < n_rovers; r++) {
                        rd.rovers.at(r).rover_step();
                    }
                    for (int r = 0; r < n_rovers; r++) {
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        for (int p = 0; p < n_poi; p++) {
                            rd.pois.at(p).observer_distances.at(r) = rd.rovers.at(r).poi_distances.at(p);
                        }
                        rd.rovers.at(r).get_nn_outputs();
                        reward = away_poi_reward(r, rd.pois);
                        local_rewards.push_back(reward);
                    }
                    rover_rewards.push_back(local_rewards);

                }

                for (int r = 0; r < n_rovers; r++) {
                    policy_id = pops.at(r).team_selection.at(t);
                    reward = accumulate(rover_rewards.at(r).begin(), rover_rewards.at(r).end(), 0.0);
                    pops.at(r).fitness.at(policy_id) = reward;
                }
            }

            for (int r = 0; r < n_rovers; r++) {
                pops.at(r).down_select();
            }
        }

        for (int rov_id = 0; rov_id < n_rovers; rov_id++) {
            nn_weights = pops.at(rov_id).population.at(0);
            save_best_policies(rov_id, s, nn_weights, fname);
        }
    }
}

void train_toward_teammates(){
    double reward;
    int policy_id;
    vector <double> nn_weights;
    vector < vector <double> > rover_rewards;
    vector <double> local_rewards;
    string fname = "TowardRoverPolicy";

    RoverDomain rd;
    rd.load_world();
    rd.load_rover();

    // Initialize CCEA population for each rover
    vector <Ccea> pops;
    for (int r = 0; r < n_rovers; r++){
        Ccea ea;
        pops.push_back(ea);
    }

    cout << "Training Go Towards Teammates Policy" << endl;
    for (int s = 0; s < stat_runs; s++) {
        cout << "Stat Run: " << s << endl;
        for (int r = 0; r < n_rovers; r++) {
            pops.at(r).create_new_population();
            rd.rovers.at(r).reset_rover();
        }

        for (int gen = 0; gen < generations; gen++) {
            for (int r = 0; r < n_rovers; r++) {
                pops.at(r).select_policy_teams();
            }

            // Test randomly selected teams
            for (int t = 0; t < pop_size; t++) {
                for (int r = 0; r < n_rovers; r++) {
                    rd.rovers.at(r).reset_rover();
                }

                // Get rover control weights and conduct initial scan
                for (int r = 0; r < n_rovers; r++) {
                    policy_id = pops.at(r).team_selection.at(t);
                    nn_weights = pops.at(r).population.at(policy_id);
                    rd.rovers.at(r).get_weights(nn_weights);
                    rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                    rd.rovers.at(r).get_nn_outputs();
                }

                // Rovers move for number of time steps
                rover_rewards.clear();
                for (int stp = 0; stp < rover_steps; stp++) {
                    local_rewards.clear();
                    for (int r = 0; r < n_rovers; r++) {
                        rd.rovers.at(r).rover_step();
                    }
                    for (int r = 0; r < n_rovers; r++) {
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        for (int p = 0; p < n_poi; p++) {
                            rd.pois.at(p).observer_distances.at(r) = rd.rovers.at(r).poi_distances.at(p);
                        }
                        rd.rovers.at(r).get_nn_outputs();
                        reward = toward_teammates_reward(r, rd.rovers);
                        local_rewards.push_back(reward);
                    }
                    rover_rewards.push_back(local_rewards);

                }

                for (int r = 0; r < n_rovers; r++) {
                    policy_id = pops.at(r).team_selection.at(t);
                    reward = accumulate(rover_rewards.at(r).begin(), rover_rewards.at(r).end(), 0.0);
                    pops.at(r).fitness.at(policy_id) = reward;
                }
            }

            for (int r = 0; r < n_rovers; r++) {
                pops.at(r).down_select();
            }
        }

        for (int rov_id = 0; rov_id < n_rovers; rov_id++) {
            nn_weights = pops.at(rov_id).population.at(0);
            save_best_policies(rov_id, s, nn_weights, fname);
        }
    }
}

void train_away_teammates(){
    double reward;
    int policy_id;
    vector <double> nn_weights;
    vector < vector <double> > rover_rewards;
    vector <double> local_rewards;
    string fname = "AwayRoversPolicy";

    RoverDomain rd;
    rd.load_world();
    rd.load_rover();

    // Initialize CCEA population for each rover
    vector <Ccea> pops;
    for (int r = 0; r < n_rovers; r++){
        Ccea ea;
        pops.push_back(ea);
    }

    cout << "Training Go Away From Teammates Policy" << endl;
    for (int s = 0; s < stat_runs; s++) {
        cout << "Stat Run: " << s << endl;
        for (int r = 0; r < n_rovers; r++) {
            pops.at(r).create_new_population();
            rd.rovers.at(r).reset_rover();
        }

        for (int gen = 0; gen < generations; gen++) {
            for (int r = 0; r < n_rovers; r++) {
                pops.at(r).select_policy_teams();
            }

            // Test randomly selected teams
            for (int t = 0; t < pop_size; t++) {
                for (int r = 0; r < n_rovers; r++) {
                    rd.rovers.at(r).reset_rover();
                }

                // Get rover control weights and conduct initial scan
                for (int r = 0; r < n_rovers; r++) {
                    policy_id = pops.at(r).team_selection.at(t);
                    nn_weights = pops.at(r).population.at(policy_id);
                    rd.rovers.at(r).get_weights(nn_weights);
                    rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                    rd.rovers.at(r).get_nn_outputs();
                }

                // Rovers move for number of time steps
                rover_rewards.clear();
                for (int stp = 0; stp < rover_steps; stp++) {
                    local_rewards.clear();
                    for (int r = 0; r < n_rovers; r++) {
                        rd.rovers.at(r).rover_step();
                    }
                    for (int r = 0; r < n_rovers; r++) {
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        for (int p = 0; p < n_poi; p++) {
                            rd.pois.at(p).observer_distances.at(r) = rd.rovers.at(r).poi_distances.at(p);
                        }
                        rd.rovers.at(r).get_nn_outputs();
                        reward = away_teammates_reward(r, rd.rovers);
                        local_rewards.push_back(reward);
                    }
                    rover_rewards.push_back(local_rewards);

                }

                for (int r = 0; r < n_rovers; r++) {
                    policy_id = pops.at(r).team_selection.at(t);
                    reward = accumulate(rover_rewards.at(r).begin(), rover_rewards.at(r).end(), 0.0);
                    pops.at(r).fitness.at(policy_id) = reward;
                }
            }

            for (int r = 0; r < n_rovers; r++) {
                pops.at(r).down_select();
            }
        }

        for (int rov_id = 0; rov_id < n_rovers; rov_id++) {
            nn_weights = pops.at(rov_id).population.at(0);
            save_best_policies(rov_id, s, nn_weights, fname);
        }
    }
}

void train_four_quadrants(int target_quadrant){
    double reward;
    int policy_id;
    vector <double> nn_weights;
    vector < vector <double> > rover_rewards;
    vector <double> local_rewards;
    string fname = "TowardQuadrant" + int_to_str(target_quadrant) + "Policy";

    RoverDomain rd;
    rd.load_world();
    rd.load_rover();

    // Initialize CCEA population for each rover
    vector <Ccea> pops;
    for (int r = 0; r < n_rovers; r++){
        Ccea ea;
        pops.push_back(ea);
    }

    cout << "Training Go Towards Quadrant: " << target_quadrant << " Policy" << endl;
    for (int s = 0; s < stat_runs; s++){
        cout << "Stat Run: " << s << endl;
        for (int r = 0; r < n_rovers; r++){
            pops.at(r).create_new_population();
            rd.rovers.at(r).reset_rover();
        }

        for (int gen = 0; gen < generations; gen++){
            for (int r = 0; r < n_rovers; r++){
                pops.at(r).select_policy_teams();
            }

            // Test randomly selected teams
            for (int t = 0; t < pop_size; t++){
                for (int r = 0; r < n_rovers; r++){
                    rd.rovers.at(r).reset_rover();
                }

                // Get rover control weights and conduct initial scan
                for (int r = 0; r < n_rovers; r++){
                    policy_id = pops.at(r).team_selection.at(t);
                    nn_weights = pops.at(r).population.at(policy_id);
                    rd.rovers.at(r).get_weights(nn_weights);
                    rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                    rd.rovers.at(r).get_nn_outputs();
                }

                // Rovers move for number of time steps
                rover_rewards.clear();
                for (int stp = 0; stp < rover_steps; stp++){
                    local_rewards.clear();
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).rover_step();
                    }
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        for (int p = 0; p < n_poi; p++){
                            rd.pois.at(p).observer_distances.at(r) = rd.rovers.at(r).poi_distances.at(p);
                        }
                        rd.rovers.at(r).get_nn_outputs();
                        reward = four_quadrant_rewards(r, rd.pois, target_quadrant);
                        local_rewards.push_back(reward);
                    }
                    rover_rewards.push_back(local_rewards);

                }

                for (int r = 0; r < n_rovers; r++){
                    policy_id = pops.at(r).team_selection.at(t);
                    reward = accumulate(rover_rewards.at(r).begin(), rover_rewards.at(r).end(), 0.0);
                    pops.at(r).fitness.at(policy_id) = reward;
                }
            }

            for (int r = 0; r < n_rovers; r++){
                pops.at(r).down_select();
            }
        }

        for (int rov_id = 0; rov_id < n_rovers; rov_id++){
            nn_weights = pops.at(rov_id).population.at(0);
            save_best_policies(rov_id, s, nn_weights, fname);
        }
    }

}

void train_two_poi(int target_poi){
    double reward;
    int policy_id;
    int n_weights = ((n_inputs + 1) * n_hidden) + ((n_hidden + 1) * n_outputs);
    vector <double> nn_weights;
    vector < vector <double> > rover_rewards;
    vector <double> local_rewards;
    string fname = "TowardPoi" + int_to_str(target_poi) + "Policy";

    RoverDomain rd;
    rd.load_world();
    rd.load_rover();

    // Initialize CCEA population for each rover
    vector <Ccea> pops;
    for (int r = 0; r < n_rovers; r++){
        Ccea ea;
        pops.push_back(ea);
    }

    cout << "Training Go Towards POI: " << target_poi << " Policy" << endl;
    for (int s = 0; s < stat_runs; s++){
        cout << "Stat Run: " << s << endl;
        for (int r = 0; r < n_rovers; r++){
            pops.at(r).create_new_population();
            rd.rovers.at(r).reset_rover();
        }

        for (int gen = 0; gen < generations; gen++){
            for (int r = 0; r < n_rovers; r++){
                pops.at(r).select_policy_teams();
            }

            // Test randomly selected teams
            for (int t = 0; t < pop_size; t++){
                for (int r = 0; r < n_rovers; r++){
                    rd.rovers.at(r).reset_rover();
                }

                // Get rover control weights and conduct initial scan
                for (int r = 0; r < n_rovers; r++){
                    policy_id = pops.at(r).team_selection.at(t);
                    nn_weights = pops.at(r).population.at(policy_id);
                    rd.rovers.at(r).get_weights(nn_weights);
                    rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                    rd.rovers.at(r).get_nn_outputs();
                }

                // Rovers move for number of time steps
                rover_rewards.clear();
                for (int stp = 0; stp < rover_steps; stp++){
                    local_rewards.clear();
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).rover_step();
                    }
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        for (int p = 0; p < n_poi; p++){
                            rd.pois.at(p).observer_distances.at(r) = rd.rovers.at(r).poi_distances.at(p);
                        }
                        rd.rovers.at(r).get_nn_outputs();
                        reward = two_poi_reward(r, rd.pois, target_poi);
                        local_rewards.push_back(reward);
                    }
                    rover_rewards.push_back(local_rewards);

                }

                for (int r = 0; r < n_rovers; r++){
                    policy_id = pops.at(r).team_selection.at(t);
                    reward = accumulate(rover_rewards.at(r).begin(), rover_rewards.at(r).end(), 0.0);
                    pops.at(r).fitness.at(policy_id) = reward;
                }
            }

            for (int r = 0; r < n_rovers; r++){
                pops.at(r).down_select();
            }
        }

        for (int rov_id = 0; rov_id < n_rovers; rov_id++){
            nn_weights = pops.at(rov_id).population.at(0);
            for (int w = 0; w < n_weights; w++){
            }
            save_best_policies(rov_id, s, nn_weights, fname);
        }
    }

}

int main(){
    cout << "Train Policy Bank" << endl;
    srand(time(NULL));
    if (policy_bank_type == "Two_Poi"){
        train_two_poi(0);
        train_two_poi(1);
    }
    else if (policy_bank_type == "Four_Quadrants"){
        for (int i = 0; i < 4; i++){
            train_four_quadrants(i);
        }
    }
    else{
        cout << "INCORRECT POLICY BANK TYPE" << endl;
    }

    return 0;
}
