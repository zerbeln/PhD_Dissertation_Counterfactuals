%Rover Domain Graphing

clear all; close all; clc

%% Test Parameters
nrovers = 12;
npoi = 10;
stat_runs = 1;
generations = 100;
coupling = 3;

%% Input from Text Files

g_reward_data = importdata('Global/Output_Data/Global_Reward.csv');
d_reward_data = importdata('Difference/Output_Data/Difference_Reward.csv');
dpp_reward_data = importdata('D++/Output_Data/DPP_Reward.csv');
sdpp_reward_data = importdata('SD++/Output_Data/SDPP_Reward.csv');

%% Data Analysis

g_fitness = mean(g_reward_data.data);
d_fitness = mean(d_reward_data.data);
dpp_fitness = mean(dpp_reward_data.data);
sdpp_fitness = mean(sdpp_reward_data.data);

%% Graph Generator

X_axis = [1:generations];
plot(X_axis, g_fitness)
hold on
plot(X_axis, d_fitness)
plot(X_axis, dpp_fitness)
plot(X_axis, sdpp_fitness)
legend('Global', 'Differnece', 'DPP', 'SDPP')
title('12 Rovers, 10 Random POIs, Fixed POI Values, 10 Stat Runs, Concentrated Rover Start, Coupling 3')
xlabel('Generations')
ylabel('System Reward')
