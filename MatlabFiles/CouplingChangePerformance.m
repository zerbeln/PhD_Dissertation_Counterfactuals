%Rover Domain Graphing

clear all; close all; clc

%% Test Parameters
nrovers = 3;
npoi = 2;
stat_runs = 40;
generations = 3000;

%% Input from files
% Original
g_loose_o = importdata('Original/LooselyCoupled/Global/Output_Data/Final_GlobalRewards.csv');
d_loose_o = importdata('Original/LooselyCoupled/Difference/Output_Data/Final_GlobalRewards.csv');
g_tight_o = importdata('Original/TightlyCoupled/Global/Output_Data/Final_GlobalRewards.csv');
d_tight_o = importdata('Original/TightlyCoupled/Difference/Output_Data/Final_GlobalRewards.csv');
dpp_orig = importdata('Original/TightlyCoupled/D++/Output_Data/Final_GlobalRewards.csv');
cba_loose_o = importdata('Original/LooselyCoupled/CBA/Output_Data/Final_GlobalRewards.csv');
cba_tight_o = importdata('Original/TightlyCoupled/CBA/Output_Data/Final_GlobalRewards.csv');

% Coupling Change (C1 -> C3)
g_c1c3 = importdata('CouplingChange/C1toC3/Global/Output_Data/Final_GlobalRewards.csv');
d_c1c3 = importdata('CouplingChange/C1toC3/Difference/Output_Data/Final_GlobalRewards.csv');
cba_c1c3 = importdata('CouplingChange/C1toC3/CBA/Output_Data/Final_GlobalRewards.csv');

% Coupling Change (C3 -> Mixed)
g_mixed = importdata('CouplingChange/C3toMixed/Global/Output_Data/Final_GlobalRewards.csv');
d_mixed = importdata('CouplingChange/C3toMixed/Difference/Output_Data/Final_GlobalRewards.csv');
dpp_mixed = importdata('CouplingChange/C3toMixed/D++/Output_Data/Final_GlobalRewards.csv');
cba_mixed = importdata('CouplingChange/C3toMixed/CBA/Output_Data/Final_GlobalRewards.csv');

% Coupling Change (C3 -> C4)
g_c3c4 = importdata('CouplingChange/C3toC4/Global/Output_Data/Final_GlobalRewards.csv');
d_c3c4 = importdata('CouplingChange/C3toC4/Difference/Output_Data/Final_GlobalRewards.csv');
dpp_c3c4 = importdata('CouplingChange/C3toC4/D++/Output_Data/Final_GlobalRewards.csv');
cba_c3c4 = importdata('CouplingChange/C3toC4/CBA/Output_Data/Final_GlobalRewards.csv');

%% Data Analysis
% Original Data Analysis
gf_L0 = mean(g_loose_o.data);
ge_L0 = std(g_loose_o.data)/stat_runs;
gf_T0 = mean(g_tight_o.data);
ge_T0 = std(g_tight_o.data)/stat_runs;
df_L0 = mean(d_loose_o.data);
de_L0 = std(d_loose_o.data)/stat_runs;
df_T0 = mean(d_tight_o.data);
de_T0 = std(d_tight_o.data)/stat_runs;
dppf0 = mean(dpp_orig.data);
dppe0 = std(dpp_orig.data)/stat_runs;
cbaf_L0 = mean(cba_loose_o);
cbae_L0 = std(cba_loose_o)/stat_runs;
cbaf_T0 = mean(cba_tight_o);
cbae_T0 = std(cba_tight_o)/stat_runs;

% Coupling Change (C1 -> C3)
gf_c1c3 = mean(g_c1c3.data);
ge_c1c3 = std(g_c1c3.data)/stat_runs;
df_c1c3 = mean(d_c1c3.data);
de_c1c3 = std(d_c1c3.data)/stat_runs;
cbaf_c1c3 = mean(cba_c1c3);
cbae_c1c3 = std(cba_c1c3)/stat_runs;

% Coupling Change (C3 -> Mixed)
gf_mix = mean(g_mixed.data);
ge_mix = std(g_mixed.data)/stat_runs;
df_mix = mean(d_mixed.data);
de_mix = std(d_mixed.data)/stat_runs;
dppf_mix = mean(dpp_mixed.data);
dppe_mix = std(dpp_mixed.data)/stat_runs;
cbaf_mix = mean(cba_mixed);
cbae_mix = std(cba_mixed)/stat_runs;

% Coupling Change (C3 -> C4)
gf_c3c4 = mean(g_c3c4.data);
ge_c3c4 = std(g_c3c4.data)/stat_runs;
df_c3c4 = mean(d_c3c4.data);
de_c3c4 = std(d_c3c4.data)/stat_runs;
dppf_c3c4 = mean(dpp_c3c4.data);
dppe_c3c4 = std(dpp_c3c4.data)/stat_runs;
cbaf_c3c4 = mean(cba_c3c4);
cbae_c3c4 = std(cba_c3c4)/stat_runs;

%% Graph Generator
color1 = [114, 147, 203]/255;
color2 = [132, 186, 91]/255;
color3 = [211, 94, 96]/255;
color5 = [128, 133, 133]/255;
color4 = [144, 103, 167]/255;
colors = [color1; color2; color3; color4; color5];
alpha = 0.3;


% C1 -> C3 --------------------------------------------------------------
ydata = [df_L0; cbaf_L0; df_c1c3; cbaf_c1c3];
ydt = ydata';
edata = [de_L0; cbae_L0; de_c1c3; cbae_c1c3];
edt = edata';

% Create Bar Plot
b = bar(ydata', 'grouped');

% Error Bars
hold on
errorbar(ydata, edata, 'k', 'linewidth', 2.0, 'linestyle', 'none');
xlabel('Fitness Type')
ylabel('Average Team Fitness')
set(gca, 'xticklabel', {'D Original', 'CBA Original', 'D C1C3', 'CBA C1C3'})

% C3 -> Mixed and C3 -> C4 ----------------------------------------------
d_data = [df_T0; df_mix; df_c3c4];
d_error = [de_T0; de_mix; de_c3c4];
dpp_data = [dppf0, dppf_mix, dppf_c3c4];
dpp_error = [dppe0, dppe_mix, dppe_c3c4];
cba_data = [cbaf_T0, cbaf_mix, cbaf_c3c4];
cba_error = [cbae_T0, cbae_mix, cbae_c3c4];
X = [1, 2, 3];

figure()
plot(X, dpp_data, '-o', 'Color', color1, 'Linewidth', 1.5)
hold on
plot(X, d_data, '-^', 'Color', color2, 'Linewidth', 1.5)
plot(X, cba_data, '-*', 'Color', color3, 'Linewidth', 1.5)

xlabel('Fitness Type')
ylabel('Average Team Fitness')
set(gca, 'xticklabel', {'Original', 'Mixed', 'C4'})
xticks([1, 2, 3])
legend('D++', 'Difference', 'CBA', 'Orientation', 'horizontal')

