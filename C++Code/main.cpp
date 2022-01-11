#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <algorithm>
#include "rover_domain.h"
#include "ccea.h"
#include "parameters.h"
#include "rewards.h"

using namespace std;

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

string int_to_str(int x) {
    stringstream ss;
    ss << x;
    return ss.str();
}

void save_best_policies(int rover_id, vector <double> policy){
    ofstream txtfile;

    string filepath = "CPP_Policy_Bank/RoverPolicy" + int_to_str(rover_id) + ".txt";
    txtfile.open(filepath);
    int vec_size = policy.size();

    for (int i = 0; i < vec_size; i++){
        txtfile << policy.at(i) << ",";
    }
    txtfile.close();
}

void rover_global_rewards(){
    double g_reward;
    int policy_id;
    vector <double> reward_history;
    vector <double> nn_weights;
    vector <double> rover_rewards;

    RoverDomain rd;
    rd.load_world();
    rd.load_rover();

    // Initialize CCEA population for each rover
    vector <Ccea> pops;
    for (int r = 0; r < n_rovers; r++){
        Ccea ea;
        pops.push_back(ea);
    }

    for (int s = 0; s < stat_runs; s++){
        cout << "Stat Run: " << s << endl;
        for (int r = 0; r < n_rovers; r++){
            pops.at(r).create_new_population();
            rd.rovers.at(r).reset_rover();
        }
        reward_history.clear();

        for (int gen = 0; gen < generations; gen++){
            //cout << "Gen: " << gen << endl;

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
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).rover_step();
                    }
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        for (int p = 0; p < n_poi; p++){
                            rd.pois.at(p).observer_distances.at(r) = rd.rovers.at(r).poi_distances.at(p);
                        }
                        rd.rovers.at(r).get_nn_outputs();
                    }
                    g_reward = rd.calc_global();
                    rover_rewards.push_back(g_reward);
                }

                for (int r = 0; r < n_rovers; r++){
                    policy_id = pops.at(r).team_selection.at(t);
                    pops.at(r).fitness.at(policy_id) = accumulate(rover_rewards.begin(), rover_rewards.end(), 0.0);
                }
            }

            // Test Best Policy
            if (gen % sample_rate == 0 or gen == generations-1){
                for (int r = 0; r < n_rovers; r++){
                    rd.rovers.at(r).reset_rover();
                }

                // Get rover control weights and conduct initial scan
                for (int r = 0; r < n_rovers; r++){
                    policy_id = int(max_element(pops.at(r).fitness.begin(), pops.at(r).fitness.end()) - pops.at(r).fitness.begin());
                    nn_weights = pops.at(r).population.at(policy_id);
                    rd.rovers.at(r).get_weights(nn_weights);
                    rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                    rd.rovers.at(r).get_nn_outputs();
                }

                // Rovers move for number of time steps
                rover_rewards.clear();
                for (int stp = 0; stp < rover_steps; stp++){
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).rover_step();
                    }
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        for (int p = 0; p < n_poi; p++){
                            rd.pois.at(p).observer_distances.at(r) = rd.rovers.at(r).poi_distances.at(p);
                        }
                        rd.rovers.at(r).get_nn_outputs();
                    }
                    g_reward = rd.calc_global();
                    rover_rewards.push_back(g_reward);
                }
                reward_history.push_back(accumulate(rover_rewards.begin(), rover_rewards.end(), 0.0));
            }

            for (int r = 0; r < n_rovers; r++){
                pops.at(r).down_select();
            }
        }

        record_reward_history(reward_history);
        for (int rov_id = 0; rov_id < n_rovers; rov_id++){
            nn_weights = pops.at(rov_id).population.at(0);
            save_best_policies(rov_id, nn_weights);
        }
    }

}


void rover_difference_rewards(){
    double g_reward;
    int policy_id;
    vector <double> reward_history;
    vector <double> nn_weights;
    vector < vector <double> > rover_rewards;
    vector <double> test_rewards;
    vector <double> difference_rewards;

    RoverDomain rd;
    rd.load_world();
    rd.load_rover();

    // Initialize CCEA population for each rover
    vector <Ccea> pops;
    for (int r = 0; r < n_rovers; r++){
        Ccea ea;
        pops.push_back(ea);
    }

    for (int s = 0; s < stat_runs; s++){
        cout << "Stat Run: " << s << endl;
        for (int r = 0; r < n_rovers; r++){
            pops.at(r).create_new_population();
            rd.rovers.at(r).reset_rover();
        }
        reward_history.clear();

        for (int gen = 0; gen < generations; gen++){
            // cout << "Gen: " << gen << endl;

            for (int r = 0; r < n_rovers; r++){
                pops.at(r).select_policy_teams();
            }

            // Test randomly selected teams
            for (int t = 0; t < pop_size; t++){
                for (int r = 0; r < n_rovers; r++){
                    rd.rovers.at(r).reset_rover();
                    pops.at(r).reset_fitness();
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
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).rover_step();
                    }
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        rd.rovers.at(r).get_nn_outputs();
                    }
                    g_reward = rd.calc_global();
                    difference_rewards = calc_difference(rd.pois, g_reward);
                    rover_rewards.push_back(difference_rewards);
                }

                for (int r = 0; r < n_rovers; r++){
                    policy_id = pops.at(r).team_selection.at(t);
                    for (int stp = 0; stp < rover_steps; stp++){
                        pops.at(r).fitness.at(policy_id) += rover_rewards.at(stp).at(r);
                    }
                }
            }

            // Test Best Policy
            if (gen % sample_rate == 0 or gen == generations-1){
                for (int r = 0; r < n_rovers; r++){
                    rd.rovers.at(r).reset_rover();
                }

                // Get rover control weights and conduct initial scan
                for (int r = 0; r < n_rovers; r++){
                    policy_id = int(max_element(pops.at(r).fitness.begin(), pops.at(r).fitness.end()) - pops.at(r).fitness.begin());
                    nn_weights = pops.at(r).population.at(policy_id);
                    rd.rovers.at(r).get_weights(nn_weights);
                    rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                    rd.rovers.at(r).get_nn_outputs();
                }

                // Rovers move for number of time steps
                test_rewards.clear();
                for (int stp = 0; stp < rover_steps; stp++){
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).rover_step();
                    }
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        rd.rovers.at(r).get_nn_outputs();
                    }
                    g_reward = rd.calc_global();
                    test_rewards.push_back(g_reward);
                }
                reward_history.push_back(accumulate(test_rewards.begin(), test_rewards.end(), 0.0));
            }

            for (int r = 0; r < n_rovers; r++){
                pops.at(r).down_select();
            }
        }
        record_reward_history(reward_history);
        for (int rov_id = 0; rov_id < n_rovers; rov_id++){
            nn_weights = pops.at(rov_id).population.at(0);
            save_best_policies(rov_id, nn_weights);
        }
    }
}


void rover_dpp_rewards(){
    double g_reward;
    int policy_id;
    vector <double> reward_history;
    vector <double> nn_weights;
    vector < vector <double> > rover_rewards;
    vector <double> test_rewards;
    vector <double> dpp_rewards;

    RoverDomain rd;
    rd.load_world();
    rd.load_rover();

    // Initialize CCEA population for each rover
    vector <Ccea> pops;
    for (int r = 0; r < n_rovers; r++){
        Ccea ea;
        pops.push_back(ea);
    }

    for (int s = 0; s < stat_runs; s++){
        cout << "Stat Run: " << s << endl;
        for (int r = 0; r < n_rovers; r++){
            pops.at(r).create_new_population();
            rd.rovers.at(r).reset_rover();
        }
        reward_history.clear();

        for (int gen = 0; gen < generations; gen++){
            // cout << "Gen: " << gen << endl;

            for (int r = 0; r < n_rovers; r++){
                pops.at(r).select_policy_teams();
            }

            // Test randomly selected teams
            for (int t = 0; t < pop_size; t++){
                for (int r = 0; r < n_rovers; r++){
                    rd.rovers.at(r).reset_rover();
                    pops.at(r).reset_fitness();
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
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).rover_step();
                    }
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        rd.rovers.at(r).get_nn_outputs();
                    }
                    g_reward = rd.calc_global();
                    dpp_rewards = calc_dpp(rd.pois, g_reward);
                    rover_rewards.push_back(dpp_rewards);
                }

                for (int r = 0; r < n_rovers; r++){
                    policy_id = pops.at(r).team_selection.at(t);
                    for (int stp = 0; stp < rover_steps; stp++){
                        pops.at(r).fitness.at(policy_id) += rover_rewards.at(stp).at(r);
                    }
                }
            }

            // Test Best Policy
            if (gen % sample_rate == 0 or gen == generations-1){
                for (int r = 0; r < n_rovers; r++){
                    rd.rovers.at(r).reset_rover();
                }

                // Get rover control weights and conduct initial scan
                for (int r = 0; r < n_rovers; r++){
                    policy_id = int(max_element(pops.at(r).fitness.begin(), pops.at(r).fitness.end()) - pops.at(r).fitness.begin());
                    nn_weights = pops.at(r).population.at(policy_id);
                    rd.rovers.at(r).get_weights(nn_weights);
                    rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                    rd.rovers.at(r).get_nn_outputs();
                }

                // Rovers move for number of time steps
                test_rewards.clear();
                for (int stp = 0; stp < rover_steps; stp++){
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).rover_step();
                    }
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        rd.rovers.at(r).get_nn_outputs();
                    }
                    g_reward = rd.calc_global();
                    test_rewards.push_back(g_reward);
                }
                reward_history.push_back(accumulate(test_rewards.begin(), test_rewards.end(), 0.0));
            }

            for (int r = 0; r < n_rovers; r++){
                pops.at(r).down_select();
            }
        }
        record_reward_history(reward_history);
        for (int rov_id = 0; rov_id < n_rovers; rov_id++){
            nn_weights = pops.at(rov_id).population.at(0);
            save_best_policies(rov_id, nn_weights);
        }
    }
}

int main() {
    if (reward_type == "Global"){
        cout << "GLOBAL REWARDS" << endl;
        rover_global_rewards();
    }
    else if (reward_type == "Difference"){
        cout << "DIFFERENCE REWARDS" << endl;
        rover_difference_rewards();
    }
    else if (reward_type == "DPP"){
        cout << "D++ REWARDS" << endl;
        rover_dpp_rewards();
    }

    return 0;
}
