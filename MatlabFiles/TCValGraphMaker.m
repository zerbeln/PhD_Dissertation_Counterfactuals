%Rover Domain Graphing

clear all; close all; clc

%% Test Parameters
nrovers = 3;
npoi = 2;
stat_runs = 40;
generations = 3000;

%% Input from files
% Global Rewards Data
g_data_same = importdata('Global/Output_Data/Final_GlobalRewards.csv');

% Difference Rewards Data
d_data_same = importdata('Difference/Output_Data/Final_GlobalRewards.csv');

% D++ Rewards Data
dpp_data_same = importdata('D++/Output_Data/Final_GlobalRewards.csv');

% CBM Data
cba_data_same = importdata('CBA/Output_Data/Final_GlobalRewards.csv');

%% Data Analysis
% Global Data Analysis
gf0 = mean(g_data_same.data);
ge0 = std(g_data_same.data)/stat_runs;
 
% Difference Data Analysis
df0 = mean(d_data_same.data);
de0 = std(d_data_same.data)/stat_runs;
 
% D++ Data Analysis
dppf0 = mean(dpp_data_same.data);
dppe0 = std(dpp_data_same.data)/stat_runs;

% CBA Data Analysis
cbaf0 = mean(cba_data_same);
cbae0 = std(cba_data_same)/stat_runs;

%% Graph Generator
color1 = [114, 147, 203]/255;
color2 = [132, 186, 91]/255;
color3 = [211, 94, 96]/255;
color5 = [128, 133, 133]/255;
color4 = [144, 103, 167]/255;
colors = [color1; color2; color3; color4; color5];
alpha = 0.3;

ydata = [dppf0; df0; gf0; cbaf0];
ydt = ydata';
edata = [dppe0; de0; ge0; cbae0];
edt = edata';

% Create Bar Plot
b = bar(ydata', 'grouped');

% Error Bars
hold on
errorbar(ydata, edata, 'k', 'linewidth', 2.0, 'linestyle', 'none');

hold off
xlabel('Fitness Type')
ylabel('Average Team Fitness')
set(gca, 'xticklabel', {'D++', 'Difference', 'Global', 'CBA'})
