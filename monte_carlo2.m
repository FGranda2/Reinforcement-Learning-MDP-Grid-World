% monte_carlo: Function solving the given MDP using the on-policy Monte
%              Carlo method
%
% Inputs:
%       world:                  A structure defining the MDP to be solved
%       epsilon:                A parameter defining the 'sofeness' of the 
%                               epsilon-soft policy
%       k_epsilon:              The decay factor of epsilon per iteration
%       omega:                  Learning rate for updating Q
%       training_iterations:    Maximum number of training episodes
%       episode_length:         Maximum number of steps in each training
%                               episodes
%
% Outputs:
%       Q:                      An array containing the action value for
%                               each state-action pair 
%       policy_index:           An array summarizing the index of the
%                               optimal action index at each state
%
% --
% Control for Robotics
% AER1517 Spring 2022
% Assignment 4
%
% --
% University of Toronto Institute for Aerospace Studies
% Dynamic Systems Lab
%
% Course Instructor:
% Angela Schoellig
% schoellig@utias.utoronto.ca
%
% Teaching Assistant: 
% SiQi Zhou
% siqi.zhou@robotics.utias.utoronto.ca
% Lukas Brunke
% lukas.brunke@robotics.utias.utoronto.ca
% Adam Hall
% adam.hall@robotics.utias.utoronto.ca
%
% This script is adapted from the course on Optimal & Learning Control for
% Autonomous Robots at the Swiss Federal Institute of Technology in Zurich
% (ETH Zurich). Course Instructor: Jonas Buchli. Course Webpage:
% http://www.adrlab.org/doku.php/adrl:education:lecture:fs2015
%
% --
% Revision history
% [20.03.07, SZ]    first version
% Modified and completed by Francisco Granda
function [Q, policy_index] = ...
    monte_carlo2(world, epsilon, ...
    k_epsilon, omega, training_iterations, episode_length)
    %% Initialization
    % MDP
    mdp = world.mdp;
    gamma = mdp.gamma;

    % States
    STATES = mdp.STATES;
    ACTIONS = mdp.ACTIONS;

    % Dimensionts
    num_states = size(STATES, 2);
    num_actions = size(ACTIONS, 2);
    
    % Create object for incremental plotting of reward after each episode
    windowSize = 10; %Sets the width of the sliding window fitler used in plotting
    plotter = RewardPlotter(windowSize);
    

    % Initialize Q
    Q = zeros(num_states, num_actions);

    % [TODO] Initialize epsilon-soft policy
    % Generate greedy deterministic policy
    random_act_index = randi(num_actions, [num_states, 1]);
    policy = zeros(num_states, num_actions);
    for s = 1:1:num_states
        selected_action = random_act_index(s);
        policy(s, selected_action) = 1;
    end
   
    %% On-policy Monte Carlo Algorithm (Section 2.9.3 of [1])
    for train_loop = 1:1:training_iterations
        % To generate E-soft policy
        prob_ngreedy = epsilon/abs(num_actions);
        prob_greedy = 1 - epsilon*(1 - 1/abs(num_actions));
        %% [TODO] Generate a training episode
        % Generate random initial state index
        cur_state_index = randi([1,num_states]);
        %cur_state_index = 1;
        % Perform Rollout
        ep_ite = 0;
        R_total = 0;
        while true
            % Add episode lenght
            ep_ite = ep_ite+1;
            % Sample current epsilon-soft policy
            greedy_idx = find(policy(cur_state_index,:));
            w = prob_ngreedy*ones(1,num_actions);
            w(greedy_idx) = prob_greedy;
            action = randsample(num_actions,1,true,w);
            % Interaction with environment
            [next_state_index, ~, reward] = ...
                one_step_gw_model(world, cur_state_index, action, 1);
            % Log data for the episode
            D(1,ep_ite) = cur_state_index;
            D(2,ep_ite) = action;
            D(3,ep_ite) = reward;
            % Continue to next state in episode
            cur_state_index = next_state_index;
            % Episode Termination
            if ep_ite == episode_length || cur_state_index == ...
                    world.mdp.s_goal_index
                break
            end
        end
        % Look for unique pairs (x,u) and first occurrence in D
        [unique_pairs,first_ocurrence] = unique(D(1:2,:)',"rows");
        unique_pairs = unique_pairs';
        first_ocurrence = first_ocurrence';
        % Compute cumulative reward and update Q(x,u)
        for i = 1:1:size(unique_pairs,2)
            % Cumulative reward
            R_i = 0;
            D_size = size(D(1,first_ocurrence(i):end),2);
            exp_val = 1;
            for val = first_ocurrence(i):1:first_ocurrence(i)+D_size-1
                if val == first_ocurrence(i)
                    R_i = D(3,first_ocurrence(i));
                else
                    R_i = R_i + gamma^exp_val * D(3,val);
                    exp_val = exp_val +1;
                end
            end
            R_total = R_total + R_i;
            % Update Q(s,a)
            Q(unique_pairs(1,i),unique_pairs(2,i)) = ...
                Q(unique_pairs(1,i),unique_pairs(2,i)) + omega*...
                (R_i-Q(unique_pairs(1,i),unique_pairs(2,i))); 
        end

        %% [TODO] Update policy(s,a)
        for i = 1:1:size(unique_pairs,2)
            % Find best Policy
            [~,best_action] = max(Q(unique_pairs(1,i),1:4));
            % Update policy matrix
            policy(unique_pairs(1,i),1:4) = zeros(1,4);
            policy(unique_pairs(1,i),best_action) = 1;
        end
        %% Decrease the exploration
        % Set k_epsilon = 1 to maintain constant exploration
        epsilon = epsilon * k_epsilon;

        
        %% [TODO] Update the reward plot
        EpisodeTotalReturn = R_total; % Sum of the reward obtained during the episode
        plotter = UpdatePlot(plotter, EpisodeTotalReturn);
        drawnow;
        %pause(0.1);

        %% Decrease the exploration
        % Set k_epsilon = 1 to maintain constant exploration
        epsilon = epsilon * k_epsilon;
        
    end
    
    % Return deterministic policy for plotting
    [~, policy_index] = max(policy, [], 2);
end
