%Rover Domain Graphing

clear all; close all; clc

%% Test Parameters
nrovers = 6;
npoi = 5;
stat_runs = 30;
generations = 2000;
coupling = 6;

%% Input from Text Files

g_reward_data = importdata('Global/Output_Data/Global_Reward.csv');
d_reward_data = importdata('Difference/Output_Data/Difference_Reward.csv');
dpp_reward_data = importdata('D++/Output_Data/DPP_Reward.csv');
sdpph_reward_data = importdata('HL/Output_Data/SDPP_Reward.csv');
sdppl_reward_data = importdata('Low/Output_Data/SDPP_Reward.csv');

% glob_PoiChoice = importdata('Global/Output_Data/POI_Choice.csv');
% dif_PoiChoice = importdata('Difference/Output_Data/POI_Choice.csv');
% dpp_PoiChoice = importdata('D++/Output_Data/POI_Choice.csv');
% hv_PoiChoice = importdata('High/Output_Data/POI_Choice.csv');
% lv_PoiChoice = importdata('Low/Output_Data/POI_Choice.csv');

%% Data Analysis

g_fitness = mean(g_reward_data.data, 1);
g_error = std(g_reward_data.data, 0, 1);

d_fitness = mean(d_reward_data.data, 1);
d_error = std(d_reward_data.data, 0, 1);

dpp_fitness = mean(dpp_reward_data.data, 1);
dpp_error = std(dpp_reward_data.data, 0, 1)/sqrt(stat_runs);

sdpp_fitness_h = mean(sdpph_reward_data.data, 1);
sdpp_error_h = std(sdpph_reward_data.data, 0, 1)/sqrt(stat_runs);

sdpp_fitness_l = mean(sdppl_reward_data.data, 1);
sdpp_error_l = std(sdppl_reward_data.data, 0, 1)/sqrt(stat_runs);

%% Graph Generator
color1 = [114, 147, 203]/255;
color2 = [132, 186, 91]/255;
color3 = [211, 94, 96]/255;
color4 = [128, 133, 133]/255;
color5 = [144, 103, 167]/255;
alpha = 0.3;

X = [1:generations];
x_axis = [X, fliplr(X)];
spacing = 20;

figure(1)
hold on
% Global Reward Data
plot(X(1:spacing:end), g_fitness(1:spacing:end), '->', 'Color', color1, 'Linewidth', 1.5)

% Difference Reward Data
plot(X(1:spacing:end), d_fitness(1:spacing:end), '-^', 'Color', color2, 'Linewidth', 1.5)

% D++ Reward Data
plot(X(1:spacing:end), dpp_fitness(1:spacing:end), '-d', 'Color', color3, 'Linewidth', 1.5)
dpp_patch = fill(x_axis, [dpp_fitness+dpp_error, fliplr(dpp_fitness-dpp_error)], color3, 'HandleVisibility','off');
set(dpp_patch, 'edgecolor', 'none');
set(dpp_patch, 'FaceAlpha', alpha);

% SD++ Reward Data
plot(X(1:spacing:end), sdpp_fitness_h(1:spacing:end), '-s', 'Color', color4, 'Linewidth', 1.5)
sdpp_patch_h = fill(x_axis, [sdpp_fitness_h+sdpp_error_h, fliplr(sdpp_fitness_h-sdpp_error_h)], color4, 'HandleVisibility','off');
set(sdpp_patch_h, 'edgecolor', 'none');
set(sdpp_patch_h, 'FaceAlpha', alpha);

plot(X(1:spacing:end), sdpp_fitness_l(1:spacing:end), '-o', 'Color', color5, 'Linewidth', 1.5)
sdpp_patch_l = fill(x_axis, [sdpp_fitness_l+sdpp_error_l, fliplr(sdpp_fitness_l-sdpp_error_l)], color5, 'HandleVisibility','off');
set(sdpp_patch_l, 'edgecolor', 'none');
set(sdpp_patch_l, 'FaceAlpha', alpha);

% Graph Options
box on
legend('Global', 'Difference', 'D++', 'S1', 'S2', 'Orientation', 'horizontal')
%title('Clusters, Coupling 3')
xlabel('Generations')
ylabel('Average System Reward')
ylim([0, 2])

% bar_color1 = [114, 147, 203]/255;
% bar_color2 = [132, 186, 91]/255;
% bar_color3 = [211, 94, 96]/255;
% bar_color4 = [128, 133, 133]/255;
% bar_color5 = [144, 103, 167]/255;
% 
% bar_data = [glob_PoiChoice(:), dif_PoiChoice(:), dpp_PoiChoice(:), hv_PoiChoice(:), lv_PoiChoice(:)];
% figure()
% x_bar = [1, 2, 0];
% b = bar(x_bar, bar_data, 'grouped');
% b(1).FaceColor = bar_color1;
% b(2).FaceColor = bar_color2;
% b(3).FaceColor = bar_color3;
% b(4).FaceColor = bar_color4;
% b(5).FaceColor = bar_color5;
% set(gca,'XTickLabel',{'None', 'POI 1', 'POI 2'})
% legend('Global', 'Difference', 'D++', 'SD++ HV', 'SD++ LV')
% xlabel('Joint Action')
% ylabel('Number of Times Selected')
