%Rover Domain Graphing

clear all; close all; clc

%% Test Parameters
nrovers = 6;
npoi = 4;
stat_runs = 10;
generations = 2000;

%% Input from files

% Hazards H1
g_h1 = importdata('H1/Global/Output_Data/Final_GlobalRewards.csv');
d_h1 = importdata('H1/Difference/Output_Data/Final_GlobalRewards.csv');
dpp_h1 = importdata('H1/D++/Output_Data/Final_GlobalRewards.csv');
cba_h1 = importdata('H1/CBA/Output_Data/Final_GlobalRewards.csv');

% Hazards H2
g_h2 = importdata('H2/Global/Output_Data/Final_GlobalRewards.csv');
d_h2 = importdata('H2/Difference/Output_Data/Final_GlobalRewards.csv');
dpp_h2 = importdata('H2/D++/Output_Data/Final_GlobalRewards.csv');
cba_h2 = importdata('H2/CBA/Output_Data/Final_GlobalRewards.csv');

% Hazards H3
g_h3 = importdata('H3/Global/Output_Data/Final_GlobalRewards.csv');
d_h3 = importdata('H3/Difference/Output_Data/Final_GlobalRewards.csv');
dpp_h3 = importdata('H3/D++/Output_Data/Final_GlobalRewards.csv');
cba_h3 = importdata('H3/CBA/Output_Data/Final_GlobalRewards.csv');

%% Data Analysis
% Hazards H1
gf_h1 = mean(g_h1.data);
ge_h1 = std(g_h1.data)/stat_runs;
df_h1 = mean(d_h1.data);
de_h1 = std(d_h1.data)/stat_runs;
dppf_h1 = mean(dpp_h1.data);
dppe_h1 = std(dpp_h1.data)/stat_runs;
cbaf_h1 = mean(cba_h1);
cbae_h1 = std(cba_h1)/stat_runs;

% Hazards H2
gf_h2 = mean(g_h2.data);
ge_h2 = std(g_h2.data)/stat_runs;
df_h2 = mean(d_h2.data);
de_h2 = std(d_h2.data)/stat_runs;
dppf_h2 = mean(dpp_h2.data);
dppe_h2 = std(dpp_h2.data)/stat_runs;
cbaf_h2 = mean(cba_h2);
cbae_h2 = std(cba_h2)/stat_runs;

% Hazards H3
gf_h3 = mean(g_h3.data);
ge_h3 = std(g_h3.data)/stat_runs;
df_h3 = mean(d_h3.data);
de_h3 = std(d_h3.data)/stat_runs;
dppf_h3 = mean(dpp_h3.data);
dppe_h3 = std(dpp_h3.data)/stat_runs;
cbaf_h3 = mean(cba_h3);
cbae_h3 = std(cba_h3)/stat_runs;

%% Graph Generator
color1 = [114, 147, 203]/255;
color2 = [132, 186, 91]/255;
color3 = [211, 94, 96]/255;
color5 = [128, 133, 133]/255;
color4 = [144, 103, 167]/255;
colors = [color1; color2; color3; color4; color5];
alpha = 0.3;

g_fit = [gf_h1, gf_h2, gf_h3];
g_error = [ge_h1, ge_h2, ge_h3];
d_fit = [df_h1, df_h2, df_h3];
d_error = [de_h1, de_h2, de_h3];
dpp_fit = [dppf_h1, dppf_h2, dppf_h3];
dpp_error = [dppe_h1, dppe_h2, dppe_h3];
cba_fit = [cbaf_h1, cbaf_h2, cbaf_h3];
cba_error = [cbae_h1, cbae_h2, cbae_h3];

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
xlabel('Hazard Configuration')
ylabel('Average Team Reward')
lgd = legend('Global', 'Difference', 'D++', 'CBA');
lgd.FontSize = 12;
set(gca, 'xticklabel', {'H1', 'H2', 'H3'})
