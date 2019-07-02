%Rover Domain Graphing

clear all; close all; clc

%% Test Parameters
nrovers = 12;
npoi = 10;
stat_runs = 1;
generations = 50;
coupling = 3;

%% Input from Text Files

g_reward_data = open('Global/Output_Data/Global_Reward.csv');
d_reward_data = open('Difference/Output_Data/Difference_Reward.csv');
dpp_reward_data = open('D++/Output_Data/DPP_Reward.csv');
%sdpp_reward_data = open('SD++/Output_Data/SDPP_Reward.csv');

%% Data Analysis

g_fitness = mean(g_reward_data.data);
d_fitness = mean(d_reward_data.data);
dpp_fitness = mean(dpp_reward_data.data);
%sdpp_fitness = mean(sdpp_reward_data.data);

%% Graph Generator

X_axis = [1:generations];
plot(X_axis, g_fitness)
hold on
plot(X_axis, d_fitness)
plot(X_axis, dpp_fitness)
%Plot(X_axis, sdpp_fitness)
legend('Global', 'Differnece', 'DPP')