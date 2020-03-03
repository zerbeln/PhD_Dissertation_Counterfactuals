import numpy as np, time
#import MultiNEAT as NEAT
import mod_mem_net as mod, sys
from random import randint#, choice
#from copy import deepcopy

class Deap_param:
    def __init__(self, angle_res, is_memoried, arch_type):
        self.num_input = (360 * 2 / angle_res) + 4
        self.num_hnodes = 15
        self.num_output = 2
        if is_memoried:
            self.type_id = 'memoried'
        else: self.type_id = 'normal'
        self.arch_type = 1  # 0-Quasi GRU; #1-Quasi NTM #Determine the neural architecture
        if arch_type == 0:
            self.arch_type = 'quasi_gru'
        elif arch_type == 1:
            self.arch_type = 'quasi_ntm'
        else:
            sys.exit('Invalid choice of neural architecture')


        self.elite_fraction = 0.1
        self.crossover_prob = 0.1
        self.mutation_prob = 0.9
        self.weight_magnitude_limit = 1000000000
        self.mut_distribution = 3  # 1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s
        if is_memoried:
            self.total_num_weights = 3 * (
                self.num_hnodes * (self.num_input + 1) + self.num_hnodes * (self.num_output + 1)) + 2 * self.num_hnodes * (
                self.num_hnodes + 1) + self.num_output * (self.num_hnodes + 1) + self.num_hnodes
        else:
            #Normalize network flexibility by changing hidden nodes
            naive_total_num_weights = self.num_hnodes*(self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
            #self.total_num_weights = self.num_hnodes * (self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
            #continue
            mem_weights = 3 * (
                 self.num_hnodes * (self.num_input + 1) + self.num_hnodes * (self.num_output + 1)) + 2 * self.num_hnodes * (
                 self.num_hnodes + 1) + self.num_output * (self.num_hnodes + 1) + self.num_hnodes
            normalization_factor = int(mem_weights/naive_total_num_weights)

            #Set parameters for comparable flexibility with memoried net
            self.num_hnodes *= normalization_factor + 1
            self.total_num_weights = self.num_hnodes * (self.num_input + 1) + self.num_output * (self.num_hnodes + 1)
        print 'Num parameters: ', self.total_num_weights

class Parameters:
    def __init__(self):
            self.population_size = 100
            self.grid_row = 20
            self.grid_col = 20
            self.total_steps = 20 # Total roaming steps without goal before termination
            self.num_predator = 3
            self.num_prey = 6
            self.predator_random = 0
            self.prey_random = 1
            self.total_generations = 20000
            self.angle_res = 45
            self.observing_prob = 1
            self.coupling = 2  # Number of predators required to simultaneously observe a prey

            self.is_memoried_predator = 0
            arch_type_pred = 0 # 0-Quasi GRU; #1-Quasi NTM #Determine the neural architecture
            self.is_memoried_prey = 1
            arch_type_prey = 1 # 0-Quasi GRU; #1-Quasi NTM #Determine the neural architecture

            self.prey_speed_boost = 1
            self.periodic_poi_mode = 0
            self.period = 0

            self.arch_type = 1  # 0-Quasi GRU; #1-Quasi NTM #Determine the neural architecture
            if self.arch_type == 0:
                self.arch_type = 'quasi_gru'
            elif self.arch_type == 1:
                self.arch_type = 'quasi_ntm'
            else:
                sys.exit('Invalid choice of neural architecture')

            #Tertiary Variables (Closed for cleanliness) Open to modify
            if True:
                # GIVEN
                # DEAP/SSNE stuff
                self.use_ssne = 1
                self.use_deap = 0
                if self.use_deap or self.use_ssne:
                    self.deap_param_predator = Deap_param(self.angle_res, self.is_memoried_predator, arch_type_pred)
                    self.deap_param_prey = Deap_param(self.angle_res, self.is_memoried_prey, arch_type_prey)

                self.D_reward = 0  # D reward scheme
                self.prey_global = 0 #Prey's reward scheme
                self.wheel_action = 0

                #TERTIARY
                self.domain_setup = 0
                self.aamas_domain = 0
                self.use_neat = 0  # Use NEAT VS. Keras based Evolution module
                self.obs_dist = 2  # Observe distance (Radius of prey that predators have to be in for successful observation)
                self.coupling = 1  # Number of predators required to simultaneously observe a prey
                self.use_py_neat = 0  # Use Python implementation of NEAT
                self.sensor_avg = True  # Average distance as state input vs (min distance by default)
                self.state_representation = 1  # 1 --> Angled brackets, 2--> List of predator/preys
                self.num_evals_ccea = 3


                #EV0-NET
                self.use_hall_of_fame = 0
                self.hof_weight = 0.99
                self.leniency = 0  # Fitness calculation based on leniency vs averaging

                if self.use_neat: #Neat
                    if self.use_py_neat: #Python NEAT
                        self.py_neat_config = Py_neat_params()

                    else: #Multi-NEAT parameters
                        import MultiNEAT as NEAT
                        self.params = NEAT.Parameters()
                        self.params.PopulationSize = self.population_size
                        self.params.fs_neat = 1
                        self.params.evo_hidden = 10
                        self.params.MinSpecies = 5
                        self.params.MaxSpecies = 15
                        self.params.EliteFraction = 0.05
                        self.params.RecurrentProb = 0.2
                        self.params.RecurrentLoopProb = 0.2

                        self.params.MaxWeight = 8
                        self.params.MutateAddNeuronProb = 0.01
                        self.params.MutateAddLinkProb = 0.05
                        self.params.MutateRemLinkProb = 0.01
                        self.params.MutateRemSimpleNeuronProb = 0.005
                        self.params.MutateNeuronActivationTypeProb = 0.005

                        self.params.ActivationFunction_SignedSigmoid_Prob = 0.01
                        self.params.ActivationFunction_UnsignedSigmoid_Prob = 0.5
                        self.params.ActivationFunction_Tanh_Prob = 0.1
                        self.params.ActivationFunction_SignedStep_Prob = 0.1
                else: #Use keras
                    self.keras_evonet_hnodes = 25  # Keras based Evo-net's hidden nodes

class Py_neat_params:
    def __init__(self):

        #[Types]
        self.stagnation_type = 'DefaultStagnation'
        self.reproduction_type = 'DefaultReproduction'

        #Phenotype
        self.input_nodes = 21
        self.hidden_nodes = 0
        self.output_nodes = 19
        self.initial_connection = 'fs_neat' #['unconnected', 'fs_neat', 'fully_connected', 'partial']
        self.max_weight = 0.1
        self.min_weight = -0.1
        self.feedforward = 0
        self.activation_functions = 'sigmoid'
        self.weight_stdev = 0.1

        #genetic
        self.pop_size = 500
        self.max_fitness_threshold = 1000000
        self.prob_add_conn = 0.1
        self.prob_add_node = 0.05
        self.prob_delete_conn = 0.01
        self.prob_delete_node = 0.01
        self.prob_mutate_bias = 0.05
        self.bias_mutation_power = 1.093
        self.prob_mutate_response = 0.1
        self.response_mutation_power = 0.1
        self.prob_mutate_weight = 0.2
        self.prob_replace_weight = 0.05
        self.weight_mutation_power = 1
        self.prob_mutate_activation = 0.08
        self.prob_toggle_link = 0.05
        self.reset_on_extinction = 1

        #genotype compatibility
        self.compatibility_threshold = 3.0
        self.excess_coefficient = 1.4
        self.disjoint_coefficient = 1.3
        self.weight_coefficient = 0.7

        self.species_fitness_func = 'mean'
        self.max_stagnation = 15

        self.elitism = 1
        self.survival_threshold = 0.2

parameters = Parameters() #Create the Parameters class
tracker = mod.statistics(parameters) #Initiate tracker
gridworld = mod.Gridworld (parameters)  # Create gridworld


def test_policies(save_name = 'trajectory.csv'):
    gridworld.load_test_policies() #Load test policies from best_team folder Assumes perfect sync
    fake_team = np.zeros(parameters.num_predators_scout + parameters.num_predators_service_bot)  # Fake team for test_phase
    best_reward = -10000
    for i in range(parameters.population_size):
        reward, global_reward, trajectory_log  = run_simulation(parameters, gridworld, fake_team, is_test=True)
        if global_reward > best_reward:
            comment = [parameters.num_predators_scout, parameters.num_predators_service_bot, parameters.num_prey] + [0] * (len(trajectory_log[0])-3)
            trajectory_log = [comment] + trajectory_log
            trajectory_log = np.array(trajectory_log)
            np.savetxt(save_name, trajectory_log, delimiter=',',fmt='%10.5f')
            print reward, global_reward
            best_reward = global_reward

def best_performance_trajectory(parameters, gridworld, teams, save_name='best_performance_traj.csv'):
    trajectory_log = []
    gridworld.reset(teams)  # Reset board
    # mod.dispGrid(gridworld)
    for steps in range(parameters.total_steps):  # One training episode till goal is not reached
        ig_traj_log = []
        for id, predator in enumerate(gridworld.predator_list_scout):  # get all the action choices from the predators
            if steps == 0: predator.perceived_state = gridworld.get_state(predator, is_Scout=True)  # Update all predator's perceived state
            predator.take_action(teams[id])  # Make the predator take action using the Evo-net with given id from the population
            ig_traj_log.append(predator.position[0])
            ig_traj_log.append(predator.position[1])

        for id, predator in enumerate(gridworld.predator_list_service_bot):  # get all the action choices from the predators
            if steps == 0: predator.perceived_state = gridworld.get_state(predator, is_Scout=False)  # Update all predator's perceived state
            predator.take_action(teams[id + parameters.num_predators_scout])  # Make the predator take action using the Evo-net with given id from the population
            ig_traj_log.append(predator.position[0])
            ig_traj_log.append(predator.position[1])



        for prey in gridworld.prey_list:
            ig_traj_log.append(prey.position[0])
            ig_traj_log.append(prey.position[1])

        trajectory_log.append(np.array(ig_traj_log))
        gridworld.move()  # Move gridworld

        # mod.dispGrid(gridworld)
        # raw_input('E')

        gridworld.update_prey_observations()  # Figure out the prey observations and store all credit information
        if parameters.is_prey_move: gridworld.prey_move()

        for id, predator in enumerate(gridworld.predator_list_scout): predator.referesh(teams[id], gridworld)  # Update state and learn if applicable
        for id, predator in enumerate(gridworld.predator_list_service_bot): predator.referesh(teams[id], gridworld)  # Update state and learn if applicable

        if gridworld.check_goal_complete(): break  # If all prey's observed

    # Log final position
    ig_traj_log = []
    for predator in gridworld.predator_list_scout:
        ig_traj_log.append(predator.position[0])
        ig_traj_log.append(predator.position[1])

    for predator in gridworld.predator_list_service_bot:
        ig_traj_log.append(predator.position[0])
        ig_traj_log.append(predator.position[1])

    for prey in gridworld.prey_list:
        ig_traj_log.append(prey.position[0])
        ig_traj_log.append(prey.position[1])
    trajectory_log.append(np.array(ig_traj_log))
    rewards, global_reward = gridworld.get_reward(teams)

    #Save trajectory to file
    comment = [parameters.num_predators_scout, parameters.num_predators_service_bot, parameters.num_prey] + [0] * (
    len(trajectory_log[0]) - 3)
    trajectory_log = [comment] + trajectory_log
    trajectory_log = np.array(trajectory_log)
    np.savetxt(save_name, trajectory_log, delimiter=',', fmt='%10.5f')

def evolve(gridworld, parameters, generation, best_hof_score):
    epoch_metrics = [];
    gridworld.new_epoch_reset() #Reset initial random positions for the epoch

    # Get new genome list and fitness evaluations trackers
    for i in range(parameters.num_predator): gridworld.predator_list[i].evo_net.referesh_genome_list()
    for i in range(parameters.num_prey): gridworld.prey_list[i].evo_net.referesh_genome_list()

    #Get selection pools
    selection_pool = mod.team_selection(gridworld, parameters)
    teams = np.zeros(parameters.num_predator + parameters.num_prey).astype(int)  # Team definitions by index

    #MAIN LOOP
    for genome_ind in range(parameters.population_size * parameters.num_evals_ccea): #For evaluation

        #PICK TEAMS
        for i in range(len(teams)):
            debug_choice_bound = (len(selection_pool[i])-1) #Hack to choose 0 when no other choices exist
            if debug_choice_bound == 0:
                rand_index = 0
            else:
                rand_index = randint(0, debug_choice_bound) #Get a random index
            teams[i] = selection_pool[i][rand_index] #Pick team member from that index
            selection_pool[i] = np.delete(selection_pool[i], rand_index) #Delete that index from the selection pool

        #SIMULATION AND TRACK REWARD
        global_reward, predator_rewards, prey_rewards = run_simulation(parameters, gridworld, teams, is_hof = False) #Returns rewards for each member of the team
        epoch_metrics.append(global_reward)

        #ENCODE FITNESS BACK TO predator
        for id, predator in enumerate(gridworld.predator_list):
            predator.evo_net.fitness_evals[teams[id]].append(predator_rewards[id]) #Assign those rewards
        # ENCODE FITNESS BACK TO prey
        for id, prey in enumerate(gridworld.prey_list):
                prey.evo_net.fitness_evals[teams[parameters.num_predator + id]].append(prey_rewards[id]) #Assign those rewards

    #if generation % 25 == 0: gridworld.save_pop() #Save population periodically

    for predator in gridworld.predator_list:
        predator.evo_net.update_fitness()# Assign fitness to genomes and update HOF net
        predator.evo_net.epoch() #Run Epoch update in the population
    if not parameters.periodic_poi_mode:
        for prey in gridworld.prey_list:
            prey.evo_net.update_fitness()# Assign fitness to genomes and update HOF net
            prey.evo_net.epoch() #Run Epoch update in the population

    #HOF VS simulation
    hof_reward, hof_predator_rewards, hof_prey_rewards = run_simulation(parameters, gridworld, teams=None, is_hof = True)

    #TODO Check HOF ASSIGNMENT


    epoch_metrics = np.array(epoch_metrics)
    #avg_epoch = np.mean(epoch_metrics)
    #sd_epoch = np.std(epoch_metrics)
    max_epoch = max(epoch_metrics)

    return max_epoch, hof_reward

def run_simulation(parameters, gridworld, teams, is_test = False, is_hof = False): #Run simulation given a team and return fitness for each individuals in that team

    #if is_test: trajectory_log = []
    if is_hof == False: gridworld.reset(teams)  # Reset board and build net
    #mod.dispGrid(gridworld)

    for steps in range(parameters.total_steps):  # One training episode till goal is not reached
        if is_test: ig_traj_log = []


        for id, predator in enumerate(gridworld.predator_list):  #get all the action choices from the predators
            predator.perceived_state = gridworld.get_state(predator) #Update all predator's perceived state
            if is_test:
                predator.take_action_test()
                ig_traj_log.append(predator.position[0])
                ig_traj_log.append(predator.position[1])
            else:
                predator.take_action(is_hof) #Make the predator take action using the Evo-net with given id from the population

        for id, prey in enumerate(gridworld.prey_list):  #get all the action choices from the predators
            #print prey.visible
            if parameters.periodic_poi_mode:
                prey.periodic_visibility(steps)  # Periodic POIs
            else:
                prey.perceived_state = gridworld.get_state(prey) #Update all predator's perceived state
                if is_test:
                    prey.take_action_test()
                    ig_traj_log.append(prey.position[0])
                    ig_traj_log.append(prey.position[1])
                else:
                    prey.take_action(is_hof) #Make the predator take action using the Evo-net with given id from the population

        gridworld.move() #Move gridworld

        #mod.dispGrid(gridworld)
        #raw_input('E')
        gridworld.update_prey_observations() #Figure out the prey observations and store all credit information

    #Log final position
    if is_test:
        ig_traj_log = []
        for predator in gridworld.predator_list_scout:
            ig_traj_log.append(predator.position[0])
            ig_traj_log.append(predator.position[1])

        for predator in gridworld.predator_list_service_bot:
            ig_traj_log.append(predator.position[0])
            ig_traj_log.append(predator.position[1])

        for prey in gridworld.prey_list:
            ig_traj_log.append(prey.position[0])
            ig_traj_log.append(prey.position[1])
        trajectory_log.append(np.array(ig_traj_log))

    global_reward, predator_rewards, prey_rewards = gridworld.get_reward()

    if is_test:
        #trajectory_log = np.array(trajectory_log)
        return rewards, global_reward, trajectory_log
    return global_reward, predator_rewards, prey_rewards

if __name__ == "__main__":

    mod.dispGrid(gridworld)
    best_hof_score = 0

    for gen in range (parameters.total_generations): #Main Loop
        #curtime = time.time()
        max_epoch, hof_reward = evolve(gridworld, parameters, gen, best_hof_score) #CCEA
        tracker.add_fitness(max_epoch, gen) #Add average global performance to tracker
        tracker.add_hof_fitness(hof_reward, gen)  # Add hof global performance to tracker
        #elapsed = time.time() - curtime

        #print elapsed


        print 'Gen:', gen,'Memoried  ' if parameters.is_memoried_predator else 'Normal ', 'Memoried  ' if parameters.is_memoried_prey else 'Normal  '\
            , ' HOF Reward:', int(hof_reward),' Cumul_hof_avg:',int(tracker.hof_avg_fitness), '  Avg epoch:',int(max_epoch), 'Cumul_Avg:',int(tracker.avg_fitness)













