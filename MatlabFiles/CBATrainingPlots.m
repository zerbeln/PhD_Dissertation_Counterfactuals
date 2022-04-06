%Rover Domain Graphing

clear all; close all; clc

%% Test Parameters
n_rovers = 3;
n_poi = 2;
stat_runs = 50;
generations = 1000;
coupling = 1;

%% Input from files

% Rover Skill Compliance Data
five_r1 = importdata('CBA/Output_Data/Rover0/CBA_Rewards.csv');
five_r2 = importdata('CBA/Output_Data/Rover1/CBA_Rewards.csv');
five_r3 = importdata('CBA/Output_Data/Rover2/CBA_Rewards.csv');
five_r4 = importdata('CBA/Output_Data/Rover3/CBA_Rewards.csv');
five_r5 = importdata('CBA/Output_Data/Rover4/CBA_Rewards.csv');
five_r6 = importdata('CBA/Output_Data/Rover5/CBA_Rewards.csv');

% Rover Skill Performance Data
s1_perf = importdata('CBA/Output_Data/Skill0_Performance.csv');
s2_perf = importdata('CBA/Output_Data/Skill1_Performance.csv');
s3_perf = importdata('CBA/Output_Data/Skill2_Performance.csv');
s4_perf = importdata('CBA/Output_Data/Skill3_Performance.csv');
s5_perf = importdata('CBA/Output_Data/Skill4_Performance.csv');


%% Data Analysis
% Rover Skill Compliance (Correctly choosing skills based on CBA)
five_r1_fit = mean(five_r1, 1);
five_r2_fit = mean(five_r2, 1);
five_r3_fit = mean(five_r3, 1);
five_r4_fit = mean(five_r4, 1);
five_r5_fit = mean(five_r5, 1);
five_r6_fit = mean(five_r6, 1);

five_combined = [five_r1_fit; five_r2_fit; five_r3_fit; five_r4_fit; five_r5_fit; five_r6_fit];
two_poi = mean(five_combined, 1);

% Rover Skill Performance Analysis
r_skills = zeros(n_rovers, n_poi);

for rov = 1:n_rovers
    r_skills(rov, 1) = mean(s1_perf(:, rov), 1);
    r_skills(rov, 2) = mean(s2_perf(:, rov), 1);
    r_skills(rov, 3) = mean(s3_perf(:, rov), 1);
    r_skills(rov, 4) = mean(s4_perf(:, rov), 1);
    r_skills(rov, 5) = mean(s5_perf(:, rov), 1);
end

%% Graph Generator
color1 = [114, 147, 203]/255;
color2 = [132, 186, 91]/255;
color3 = [211, 94, 96]/255;
color4 = [128, 133, 133]/255;
color5 = [144, 103, 167]/255;
color6 = [50, 144, 147]/255;
alpha = 0.3;

X = [0:20:(generations-1)];
x_axis = [X, fliplr(X)];
spacing = 1;

% Rover Skill Compliance -------------------------------------------------
figure(1)
hold on
plot(X, five_r1_fit, '-o', 'Color', color1, 'Linewidth', 1.5)
plot(X, five_r2_fit, '-^', 'Color', color2, 'Linewidth', 1.5)
plot(X, five_r3_fit, '-*', 'Color', color3, 'Linewidth', 1.5)
plot(X, five_r4_fit, '-', 'Color', color4, 'Linewidth', 1.5)
plot(X, five_r5_fit, '--', 'Color', color5, 'Linewidth', 1.5)
plot(X, five_r6_fit, '-s', 'Color', color6, 'Linewidth', 1.5)

% Graph Options
box on
legend('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'Orientation', 'horizontal')
title('Rover Skill Compliance')
xlabel('Generations')
ylabel('Skill Selection Accuracy')


% Rover Skill Performance ------------------------------------------------
R = [1:n_rovers];

figure()
hold on
plot(R, r_skills(:, 1), '-o', 'Color', color1, 'Linewidth', 1.5)
plot(R, r_skills(:, 2), '-^', 'Color', color2, 'Linewidth', 1.5)
plot(R, r_skills(:, 3), '-*', 'Color', color3, 'Linewidth', 1.5)
plot(R, r_skills(:, 4), '-', 'Color', color4, 'Linewidth', 1.5)
plot(R, r_skills(:, 5), '--', 'Color', color5, 'Linewidth', 1.5)

xlabel('Rover ID')
set(gca, 'xticklabel', {'R1', 'R2', 'R3', 'R4', 'R5', 'R6'});
ylabel('Average PoI Reward')
legend('P1', 'P2', 'P3', 'P4', 'P5', 'Orientation', 'horizontal')
title('Rover Skill Performance')



