% generalized_policy_iteration: Function solving the given MDP using the
%                               Generalized Policy Iteration algorithm
%
% Inputs:
%       world:                  A structure defining the MDP to be solved
%       precision_pi:           Maximum value function change before
%                               terminating Policy Improvement step
%       max_ite_pi:             Maximum number of iterations for Policy
%                               Improvement loop
%       precision_pe:           Maximum value function change before
%                               terminating Policy Evaluation step
%       max_ite_pe:             Maximum number of iterations for Policy
%                               Evaluation loop
%
% Outputs:
%       V:                      An array containing the value at each state
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

function [V, policy_index] = ...
    generalized_policy_iteration(world, ...
    precision_pi, precision_pe, max_ite_pi, max_ite_pe)
    %% Initialization
    % MDP
    mdp = world.mdp;
    T = mdp.T;
    R = mdp.R;
    gamma = mdp.gamma;

    % Dimensions
    num_actions = length(T);
    num_states = size(T{1}, 1);

    % Intialize value function
    V = zeros(num_states, 1);

    % Initialize policy
    % Note: Policy here encodes the action to be executed at state s. We
    %       use deterministic policy here (e.g., [0,1,0,0] means take 
    %       action indexed 2)
    random_act_index = randi(num_actions, [num_states, 1]);
    policy = zeros(num_states, num_actions);
    for s = 1:1:num_states
        selected_action = random_act_index(s);
        policy(s, selected_action) = 1;
    end
    ite_pi = 0;
    while true
        %% [TODO] policy Evaluation (PE) (Section 2.6 of [1])
        % V = ...;
        ite_pe = 0;
        while true
            ite_pe = ite_pe+1;
            delta = 0;
            for x = 1:num_states
                % Convergence Condition
                v = V(x);
                % Find index of current policy applied at state x
                curr_policy = find(policy(x,:));
                % Compute cost for state x
                next_state = find(T{1,curr_policy}(x,:));
                V(x) = R{1,curr_policy}(x,next_state) + ...
                    gamma * V(next_state);
                % Check for convergence
                delta = max([delta,abs(v-V(x))]); 
            end
            if delta < precision_pe || ite_pe == max_ite_pe
                break
            end
        end


        %% [TODO] Policy Improvment (PI) (Section 2.7 of [1])
        % policy = ...;
        ite_pi = ite_pi+1;
        % For convergence
        b = policy;
        for x = 1:num_states           
            % Evaluate policies
            for a = 1:num_actions
                curr_policy = a;
                % Compute cost for state x
                next_state = find(T{1,curr_policy}(x,:));
                cost(1,a) = R{1,curr_policy}(x,next_state) ...
                    + gamma * V(next_state);
            end
            % Find best Policy
            [~,best_action] = max(cost);
            % Update policy matrix
            policy(x,1:4) = zeros(1,4);
            policy(x,best_action) = 1;
        end
        if ite_pi == max_ite_pi || isequal(b,policy)
            break
        end
    end
 
    % Return deterministic policy for plotting
    [~, policy_index] = max(policy, [], 2);
end