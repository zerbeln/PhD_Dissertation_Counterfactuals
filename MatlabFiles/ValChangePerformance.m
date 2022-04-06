%Rover Domain Graphing

clear all; close all; clc

%% Test Parameters
nrovers = 6;
npoi = 5;
stat_runs = 50;
generations = 2000;

%% Input from files

% Coupling Change C1
g_v1 = importdata('V1/Global/Output_Data/Final_GlobalRewards.csv');
d_v1 = importdata('V1/Difference/Output_Data/Final_GlobalRewards.csv');
dpp_v1 = importdata('V1/D++/Output_Data/Final_GlobalRewards.csv');
cba_v1 = importdata('V1/CBA/Output_Data/Final_GlobalRewards.csv');

% Coupling Change C2
g_v2 = importdata('V2/Global/Output_Data/Final_GlobalRewards.csv');
d_v2 = importdata('V2/Difference/Output_Data/Final_GlobalRewards.csv');
dpp_v2 = importdata('V2/D++/Output_Data/Final_GlobalRewards.csv');
cba_v2 = importdata('V2/CBA/Output_Data/Final_GlobalRewards.csv');

% Coupling Change C3
g_v3 = importdata('V3/Global/Output_Data/Final_GlobalRewards.csv');
d_v3 = importdata('V3/Difference/Output_Data/Final_GlobalRewards.csv');
dpp_v3 = importdata('V3/D++/Output_Data/Final_GlobalRewards.csv');
cba_v3 = importdata('V3/CBA/Output_Data/Final_GlobalRewards.csv');

%% Data Analysis
% Coupling Change C1
gf_v1 = mean(g_v1.data);
ge_v1 = std(g_v1.data)/stat_runs;
df_v1 = mean(d_v1.data);
de_v1 = std(d_v1.data)/stat_runs;
dppf_v1 = mean(dpp_v1.data);
dppe_v1 = std(dpp_v1.data)/stat_runs;
cbaf_v1 = mean(cba_v1);
cbae_v1 = std(cba_v1)/stat_runs;

% Coupling Change C2
gf_v2 = mean(g_v2.data);
ge_v2 = std(g_v2.data)/stat_runs;
df_v2 = mean(d_v2.data);
de_v2 = std(d_v2.data)/stat_runs;
dppf_v2 = mean(dpp_v2.data);
dppe_v2 = std(dpp_v2.data)/stat_runs;
cbaf_v2 = mean(cba_v2);
cbae_v2 = std(cba_v2)/stat_runs;

% Coupling Change C3
gf_v3 = mean(g_v3.data);
ge_v3 = std(g_v3.data)/stat_runs;
df_v3 = mean(d_v3.data);
de_v3 = std(d_v3.data)/stat_runs;
dppf_v3 = mean(dpp_v3.data);
dppe_v3 = std(dpp_v3.data)/stat_runs;
cbaf_v3 = mean(cba_v3);
cbae_v3 = std(cba_v3)/stat_runs;

%% Graph Generator
color1 = [114, 147, 203]/255;
color2 = [132, 186, 91]/255;
color3 = [211, 94, 96]/255;
color5 = [128, 133, 133]/255;
color4 = [144, 103, 167]/255;
colors = [color1; color2; color3; color4; color5];
alpha = 0.3;

g_fit = [gf_v1, gf_v2, gf_v3];
g_error = [ge_v1, ge_v2, ge_v3];
d_fit = [df_v1, df_v2, df_v3];
d_error = [de_v1, de_v2, de_v3];
dpp_fit = [dppf_v1, dppf_v2, dppf_v3];
dpp_error = [dppe_v1, dppe_v2, dppe_v3];
cba_fit = [cbaf_v1, cbaf_v2, cbaf_v3];
cba_error = [cbae_v1, cbae_v2, cbae_v3];

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
xlabel('PoI Value Configuration')
ylabel('Average Team Reward')
lgd = legend('Global', 'Difference', 'D++', 'CBA');
lgd.FontSize = 12;
set(gca, 'xticklabel', {'V1', 'V2', 'V3'})
