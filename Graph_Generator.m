%Rover Domain Graphing

clear all; close all; clc

%% Test Parameters
nrovers = 3;
npoi = 4;
stat_runs = 15;
generations = 3000;
coupling = 1;

%% Input from files

g_reward_data = importdata('Global/Output_Data/Global_Reward.csv');
d_reward_data = importdata('Difference/Output_Data/Difference_Reward.csv');
s_reward_data = importdata('Suggestion/Output_Data/Suggestion_GlobalReward.csv');


%% Data Analysis

g_fitness = mean(g_reward_data.data, 1);
g_error = std(g_reward_data.data, 0, 1);

d_fitness = mean(d_reward_data.data, 1);
d_error = std(d_reward_data.data, 0, 1);

s_fitness = mean(s_reward_data.data, 1);
s_error = std(s_reward_data.data, 0, 1);

%% Graph Generator
color1 = [114, 147, 203]/255;
color2 = [132, 186, 91]/255;
color3 = [211, 94, 96]/255;
color4 = [128, 133, 133]/255;
color5 = [144, 103, 167]/255;
alpha = 0.3;

X = [0:10:generations];
x_axis = [X, fliplr(X)];
spacing = 15;

figure(1)
hold on
% Global Reward Data
plot(X(1:spacing:end), g_fitness(1:spacing:end), '->', 'Color', color1, 'Linewidth', 1.5)

% Difference Reward Data
plot(X(1:spacing:end), d_fitness(1:spacing:end), '-^', 'Color', color2, 'Linewidth', 1.5)

% Suggestions
plot(X(1:spacing:end), s_fitness(1:spacing:end), '-^', 'Color', color2, 'Linewidth', 1.5)


% Graph Options
box on
legend('Global', 'Difference', 'Suggestions', 'Orientation', 'horizontal')
%title('Clusters, Coupling 3')
xlabel('Generations')
ylabel('Average System Reward')

