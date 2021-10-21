close all; clear all; clc

nrovers = 3;
npoi = 4;
generations = 2000;
stat_runs = 1;
sample_step_size = 10;

gen_x_axis = 1:10:generations;

policy_accuracy = zeros(npoi, generations/sample_step_size);
figure()
hold on
for i = 1:npoi
    fpath = sprintf('Output_Data/TargetPOI%d_Accuracy.csv', i-1);
    in_data = importdata(fpath);
    policy_accuracy(i, :) = mean(in_data.data);
    
    % Plot the accuracy of each policy
    plot(gen_x_axis, policy_accuracy(i,:), 'linewidth', 2.0)
    xlabel('Generation')
    ylabel('Percentage Accuracy')
end
title('Policy Accuracy')
legend('Policy 0', 'Policy 1', 'Policy 2', 'Policy 3')

policy_rewards = zeros(npoi, generations/sample_step_size);
figure()
hold on
for i = 1:npoi
    fpath = sprintf('Output_Data/TargetPOI%d_Rewards.csv', i-1);
    in_data = importdata(fpath);
    policy_rewards(i, :) = mean(in_data.data);
    
    % Plot the reward of each policy
    plot(gen_x_axis, policy_rewards(i,:), 'linewidth', 2.0)
    xlabel('Generation')
    ylabel('Average Reward')
end
title('Average Global Reward')
legend('Policy 0', 'Policy 1', 'Policy 2', 'Policy 3')

%% Selection Policy Accuracy
select_acc = importdata('Output_Data/SelectionAccuracy.csv');
s_accuracy = mean(select_acc.data);

figure()
plot(gen_x_axis, s_accuracy, 'linewidth', 2.0)
xlabel('Generation')
ylabel('Percentage Accuracy')
title('Suggestion Interpreter Accuracy')