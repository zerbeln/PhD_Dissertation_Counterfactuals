%Rover Domain Graphing

clear all; close all; clc

%% Test Parameters
nrovers = 6;
npoi = 5;
stat_runs = 30;
generations = 5000;

%% Input from files
% Global Rewards Data
g_data_same = importdata('Same/Global/Output_Data/Final_GlobalRewards.csv');
g_data_c1 = importdata('V1/Global/Output_Data/Final_GlobalRewards.csv');
g_data_c2 = importdata('V2/Global/Output_Data/Final_GlobalRewards.csv');
g_data_c3 = importdata('V3/Global/Output_Data/Final_GlobalRewards.csv');

% Difference Rewards Data
d_data_same = importdata('Same/Difference/Output_Data/Final_GlobalRewards.csv');
d_data_c1 = importdata('V1/Difference/Output_Data/Final_GlobalRewards.csv');
d_data_c2 = importdata('V2/Difference/Output_Data/Final_GlobalRewards.csv');
d_data_c3 = importdata('V3/Difference/Output_Data/Final_GlobalRewards.csv');

% D++ Rewards Data
dpp_data_same = importdata('Same/DPP/Output_Data/Final_GlobalRewards.csv');
dpp_data_c1 = importdata('V1/DPP/Output_Data/Final_GlobalRewards.csv');
dpp_data_c2= importdata('V2/DPP/Output_Data/Final_GlobalRewards.csv');
dpp_data_c3 = importdata('V3/DPP/Output_Data/Final_GlobalRewards.csv');

% S-D++(1) Reward Data
sdpp1_data_same = importdata('Same/SDPP1/Output_Data/Final_GlobalRewards.csv');
sdpp1_data_c1 = importdata('V1/SDPP1/Output_Data/Final_GlobalRewards.csv');
sdpp1_data_c2= importdata('V2/SDPP1/Output_Data/Final_GlobalRewards.csv');
sdpp1_data_c3 = importdata('V3/SDPP1/Output_Data/Final_GlobalRewards.csv');

% S-D++(2) Reward Data
sdpp2_data_same = importdata('Same/SDPP2/Output_Data/Final_GlobalRewards.csv');
sdpp2_data_c1 = importdata('V1/SDPP2/Output_Data/Final_GlobalRewards.csv');
sdpp2_data_c2= importdata('V2/SDPP2/Output_Data/Final_GlobalRewards.csv');
sdpp2_data_c3 = importdata('V3/SDPP2/Output_Data/Final_GlobalRewards.csv');

% CBM Data
cbm_data_same = importdata('Same/CBM/Output_Data/Final_GlobalRewards.csv');
cbm_data_c1 = importdata('V1/CBM/Output_Data/Final_GlobalRewards.csv');
cbm_data_c2 = importdata('V2/CBM/Output_Data/Final_GlobalRewards.csv');
cbm_data_c3 = importdata('V3/CBM/Output_Data/Final_GlobalRewards.csv');


%% Data Analysis
% Global Data Analysis
gf0 = mean(g_data_same.data(:));
gf1 = mean(g_data_c1.data(:));
gf2 = mean(g_data_c2.data(:));
gf3 = mean(g_data_c3.data(:));

ge0 = std(g_data_same.data(:))/stat_runs;
ge1 = std(g_data_c1.data(:))/stat_runs;
ge2 = std(g_data_c2.data(:))/stat_runs;
ge3 = std(g_data_c3.data(:))/stat_runs;
 
g_fitness = [gf1, gf2, gf3];
g_error = [ge1, ge2, ge3];
 
% Difference Data Analysis
df0 = mean(d_data_same.data(:));
df1 = mean(d_data_c1.data(:));
df2 = mean(d_data_c2.data(:));
df3 = mean(d_data_c3.data(:));

de0 = std(d_data_same.data(:))/stat_runs;
de1 = std(d_data_c1.data(:))/stat_runs;
de2 = std(d_data_c2.data(:))/stat_runs;
de3 = std(d_data_c3.data(:))/stat_runs;
 
d_fitness = [df1, df2, df3];
d_error = [de1, de2, de3];
 
% D++ Data Analysis
dppf0 = mean(dpp_data_same.data(:));
dppf1 = mean(dpp_data_c1.data(:));
dppf2 = mean(dpp_data_c2.data(:));
dppf3 = mean(dpp_data_c3.data(:));

dppe0 = std(dpp_data_same.data(:))/stat_runs;
dppe1 = std(dpp_data_c1.data(:))/stat_runs;
dppe2 = std(dpp_data_c2.data(:))/stat_runs;
dppe3 = std(dpp_data_c3.data(:))/stat_runs;

dpp_fitness = [dppf1, dppf2, dppf3];
dpp_error = [dppe1, dppe2, dppe3];

% S-D++1 Data Analysis
sdpp1f0 = mean(sdpp1_data_same.data(:));
sdpp1f1 = mean(sdpp1_data_c1.data(:));
sdpp1f2 = mean(sdpp1_data_c2.data(:));
sdpp1f3 = mean(sdpp1_data_c3.data(:));

sdpp1e0 = std(sdpp1_data_same.data(:))/stat_runs;
sdpp1e1 = std(sdpp1_data_c1.data(:))/stat_runs;
sdpp1e2 = std(sdpp1_data_c2.data(:))/stat_runs;
sdpp1e3 = std(sdpp1_data_c3.data(:))/stat_runs;

sdpp1_fitness = [sdpp1f1, sdpp1f2, sdpp1f3];
sdpp1_error = [sdpp1e1, sdpp1e2, sdpp1e3];

% S-D++2 Data Analysis
sdpp2f0 = mean(sdpp2_data_same.data(:));
sdpp2f1 = mean(sdpp2_data_c1.data(:));
sdpp2f2 = mean(sdpp2_data_c2.data(:));
sdpp2f3 = mean(sdpp2_data_c3.data(:));

sdpp2e0 = std(sdpp2_data_same.data(:))/stat_runs;
sdpp2e1 = std(sdpp2_data_c1.data(:))/stat_runs;
sdpp2e2 = std(sdpp2_data_c2.data(:))/stat_runs;
sdpp2e3 = std(sdpp2_data_c3.data(:))/stat_runs;

sdpp2_fitness = [sdpp2f1, sdpp2f2, sdpp2f3];
sdpp2_error = [sdpp2e1, sdpp2e2, sdpp2e3];

% CBM Data Analysis
sf0 = mean(cbm_data_same.data(:));
sf1 = mean(cbm_data_c1.data(:));
sf2 = mean(cbm_data_c2.data(:));
sf3 = mean(cbm_data_c3.data(:));

cbme0 = std(cbm_data_same.data(:))/stat_runs;
cbme1 = std(cbm_data_c1.data(:))/stat_runs;
cbme2 = std(cbm_data_c2.data(:))/stat_runs;
cbme3 = std(cbm_data_c3.data(:))/stat_runs;

cbm_fitness = [sf1, sf2, sf3];
cbm_error = [cbme1, cbme2, cbme3];

%% Graph Generator
color1 = [114, 147, 203]/255;
color2 = [132, 186, 91]/255;
color3 = [211, 94, 96]/255;
color5 = [128, 133, 133]/255;
color4 = [144, 103, 167]/255;
colors = [color1; color2; color3; color4; color5];
alpha = 0.3;

ydata = [g_fitness; d_fitness; dpp_fitness; cbm_fitness];
ydt = ydata';
edata = [g_error; d_error; dpp_error; cbm_error];
edt = edata';

% Create Bar Plot
b = bar(ydata', 'grouped');
for k = 1:4
    b(k).FaceColor = colors(k,:);
end

% Error Bars
hold on
[ngroups, nbars] = size(ydata');
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
    errorbar(x(i,:), ydata(i,:), edata(i,:), 'k', 'linewidth', 2.0, 'linestyle', 'none');
end

hold off
xlabel('PoI Value Configuration')
ylabel('Average Team Reward')
lgd = legend('Global', 'Difference', 'D++', 'CBM');
lgd.FontSize = 12;
set(gca, 'xticklabel', {'V1', 'V2', 'V3'})
