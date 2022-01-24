#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <algorithm>
#include "rover_domain.h"
#include "ccea.h"
#include "parameters.h"
#include "rewards.h"
#include "calc_angle_dist.h"

using namespace std;

string int_to_str(int x) {
    stringstream ss;
    ss << x;
    return ss.str();
}

vector <double> load_trained_policies(string filename, int rover_id, int srun){
    vector <double> nn_weights;
    string weight;
    int n_weights = ((n_inputs + 1) * n_hidden) + ((n_hidden + 1) * n_outputs);

    string filepath = "Policy_Bank/Rover" + int_to_str(rover_id) + "/" + filename + int_to_str(srun) + ".txt";
    cout << filepath << endl;

    ifstream txtfile(filepath, ios::in);

    for (int w = 0; w < n_weights; w++){
        getline(txtfile, weight, ',');
        nn_weights.push_back(stod(weight));

        cout << nn_weights.at(w) << " ";
    }
    cout << endl;

    return nn_weights;
}

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

void save_best_policies(int rover_id, vector <double> policy, string filename){
    ofstream txtfile;

    string filepath = "Policy_Bank/Rover" + int_to_str(rover_id) + "/" + filename + ".txt";
    txtfile.open(filepath);
    int vec_size = policy.size();

    for (int i = 0; i < vec_size; i++){
        txtfile << policy.at(i) << ",";
    }
    txtfile.close();
}

// TO DO: WRITE COUNTERFACTUAL STATE CREATION FUNCTIONS

vector <double> create_counterfactual_poi_state(vector <Poi> pois, double rx, double ry, int suggestion){
    vector <double> c_poi_state(4, 0.0);
    vector < vector <double> > temp_poi_dist_list(4);
    double angle, dist;
    int bracket, num_poi_bracket;

    for (int p = 0; p < n_poi; p++) {
        angle, dist = get_angle_dist(rx, ry, pois.at(p).x_position, pois.at(p).y_position);
        bracket = int(angle / angle_res);
        if (bracket > 3) {
            bracket -= 4;
        }

        temp_poi_dist_list.at(bracket).push_back(pois.at(p).poi_value / dist);
    }

    for (bracket = 0; bracket < 4; bracket++) {
        num_poi_bracket = temp_poi_dist_list.at(bracket).size();
        if (num_poi_bracket > 0) {
            if (sensor_model == "summed"){
                c_poi_state.at(bracket) = double(accumulate(temp_poi_dist_list.at(bracket).begin(), temp_poi_dist_list.at(bracket).end(), 0.0));
            }
            else if (sensor_model == "density"){
                c_poi_state.at(bracket) = double(accumulate(temp_poi_dist_list.at(bracket).begin(), temp_poi_dist_list.at(bracket).end(), 0.0) / num_poi_bracket);
            }
        } else {
            c_poi_state.at(bracket) = -1.0;
        }
    }

    return c_poi_state;
}

vector <double> create_counterfactual_rover_state(vector <Rover> rovers, double rx, double ry, int rover_id){
    vector <double> c_rover_state(4, 0.0);
    vector <vector <double> > temp_rover_dist_list(4);
    double angle, dist, tx, ty;
    int bracket, num_rover_bracket;

    for (int r = 0; r < n_rovers; r++) {
        if (r != rover_id){
            tx = rovers.at(r).rx;
            ty = rovers.at(r).ry;
            angle, dist = get_angle_dist(rx, ry, tx, ty);
            bracket = int(angle / angle_res);
            if (bracket > 3) {
                bracket -= 4;
            }

            temp_rover_dist_list.at(bracket).push_back(1 / dist);
        }

    }

    for (bracket = 0; bracket < 4; bracket++) {
        num_rover_bracket = temp_rover_dist_list.at(bracket).size();
        if (num_rover_bracket > 0) {
            if (sensor_model == "summed"){
                c_rover_state.at(bracket) = double(accumulate(temp_rover_dist_list.at(bracket).begin(), temp_rover_dist_list.at(bracket).end(), 0.0));
            }
            else if (sensor_model== "density"){
                c_rover_state.at(bracket) = double(accumulate(temp_rover_dist_list.at(bracket).begin(), temp_rover_dist_list.at(bracket).end(), 0.0) / num_rover_bracket);
            }
        } else {
            c_rover_state.at(bracket) = -1.0;
        }
    }

    return c_rover_state;
}

vector <double> create_counterfactual_state(vector <Rover> rovers, vector <Poi> pois, int rover_id, int suggestion){
    vector <double> c_state, r_state, p_state;
    double rx, ry;

    rx = rovers.at(rover_id).rx;
    ry = rovers.at(rover_id).ry;

    r_state = create_counterfactual_rover_state(rovers, rx, ry, rover_id);
    p_state = create_counterfactual_poi_state(pois, rx, ry, suggestion);
    c_state.reserve(r_state.size() + p_state.size());
    c_state.insert(c_state.end(), p_state.begin(), p_state.end());
    c_state.insert(c_state.end(), r_state.begin(), r_state.end());

    return c_state;
}

void train_suggestion_interpreter(){
    double g_reward;
    int policy_id;
    string filename;
    vector <double> reward_history;
    vector <double> nn_weights;
    vector <double> rover_rewards;
    vector <double> sensor_readings, c_state;

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

        // Load trained policy banks for each rover
        for (int r = 0; r < n_rovers; r++){
            filename = "TowardPoi0Policy";
            nn_weights = load_trained_policies(filename, r, s);
            rd.rovers.at(r).policy_bank.push_back(nn_weights);

            filename = "TowardPoi1Policy";
            nn_weights = load_trained_policies(filename, r, s);
            rd.rovers.at(r).policy_bank.push_back(nn_weights);
        }

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

                // Get rover control weights and conduct initial scan
                for (int r = 0; r < n_rovers; r++){
                    policy_id = pops.at(r).team_selection.at(t);
                    nn_weights = pops.at(r).population.at(policy_id);
                    rd.rovers.at(r).get_weights(nn_weights);

                }

                for (int sgst = 0; sgst < n_suggestions; sgst++){
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).reset_rover();
                    }
                    for (int r = 0; r < n_rovers; r++){
                        rd.rovers.at(r).scan_environment(rd.pois, rd.rovers);
                        c_state = create_counterfactual_state(rd.rovers, rd.pois, r, sgst);
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
                            c_state = create_counterfactual_state(rd.rovers, rd.pois, r, sgst);
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
            filename = "CBMPolicy" + int_to_str(s);
            save_best_policies(rov_id, nn_weights, filename);
        }
    }

}


int main() {

    train_suggestion_interpreter();


    return 0;
}
