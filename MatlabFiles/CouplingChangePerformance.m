%Rover Domain Graphing

clear all; close all; clc

%% Test Parameters
nrovers = 6;
npoi = 5;
stat_runs = 50;
generations = 2000;

%% Input from files

% Coupling Change C1
g_c1 = importdata('C1/Global/Output_Data/Final_GlobalRewards.csv');
d_c1 = importdata('C1/Difference/Output_Data/Final_GlobalRewards.csv');
dpp_c1 = importdata('C1/D++/Output_Data/Final_GlobalRewards.csv');
cba_c1 = importdata('C1/CBA/Output_Data/Final_GlobalRewards.csv');

% Coupling Change C2
g_c2 = importdata('C2/Global/Output_Data/Final_GlobalRewards.csv');
d_c2 = importdata('C2/Difference/Output_Data/Final_GlobalRewards.csv');
dpp_c2 = importdata('C2/D++/Output_Data/Final_GlobalRewards.csv');
cba_c2 = importdata('C2/CBA/Output_Data/Final_GlobalRewards.csv');

% Coupling Change C3
g_c3 = importdata('C3/Global/Output_Data/Final_GlobalRewards.csv');
d_c3 = importdata('C3/Difference/Output_Data/Final_GlobalRewards.csv');
dpp_c3 = importdata('C3/D++/Output_Data/Final_GlobalRewards.csv');
cba_c3 = importdata('C3/CBA/Output_Data/Final_GlobalRewards.csv');

%% Data Analysis
% Coupling Change C1
gf_c1 = mean(g_c1.data);
ge_c1 = std(g_c1.data)/stat_runs;
df_c1 = mean(d_c1.data);
de_c1 = std(d_c1.data)/stat_runs;
dppf_c1 = mean(dpp_c1.data);
dppe_c1 = std(dpp_c1.data)/stat_runs;
cbaf_c1 = mean(cba_c1);
cbae_c1 = std(cba_c1)/stat_runs;

% Coupling Change C2
gf_c2 = mean(g_c2.data);
ge_c2 = std(g_c2.data)/stat_runs;
df_c2 = mean(d_c2.data);
de_c2 = std(d_c2.data)/stat_runs;
dppf_c2 = mean(dpp_c2.data);
dppe_c2 = std(dpp_c2.data)/stat_runs;
cbaf_c2 = mean(cba_c2);
cbae_c2 = std(cba_c2)/stat_runs;

% Coupling Change C3
gf_c3 = mean(g_c3.data);
ge_c3 = std(g_c3.data)/stat_runs;
df_c3 = mean(d_c3.data);
de_c3 = std(d_c3.data)/stat_runs;
dppf_c3 = mean(dpp_c3.data);
dppe_c3 = std(dpp_c3.data)/stat_runs;
cbaf_c3 = mean(cba_c3);
cbae_c3 = std(cba_c3)/stat_runs;

%% Graph Generator
color1 = [114, 147, 203]/255;
color2 = [132, 186, 91]/255;
color3 = [211, 94, 96]/255;
color5 = [128, 133, 133]/255;
color4 = [144, 103, 167]/255;
colors = [color1; color2; color3; color4; color5];
alpha = 0.3;


g_fit = [gf_c1, gf_c2, gf_c3];
g_error = [ge_c1, ge_c2, ge_c3];
d_fit = [df_c1, df_c2, df_c3];
d_error = [de_c1, de_c2, de_c3];
dpp_fit = [dppf_c1, dppf_c2, dppf_c3];
dpp_error = [dppe_c1, dppe_c2, dppe_c3];
cba_fit = [cbaf_c1, cbaf_c2, cbaf_c3];
cba_error = [cbae_c1, cbae_c2, cbae_c3];

ydata = [g_fit; d_fit; dpp_fit; cba_fit];
ydt = ydata';
edata = [g_error; d_error; dpp_error; cba_error];
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
xlabel('PoI Coupling Configuration')
ylabel('Average Team Reward')
lgd = legend('Global', 'Difference', 'D++', 'CBA');
lgd.FontSize = 12;
set(gca, 'xticklabel', {'C1', 'C2', 'C3'})

