"""
 Solver.py

 Python function template to solve the discounted stochastic
 shortest path problem.

 Dynamic Programming and Optimal Control
 Fall 2023
 Programming Exercise
 
 Contact: Antonio Terpin aterpin@ethz.ch
 
 Authors: Abhiram Shenoi, Philip Pawlowsky, Antonio Terpin

 --
 ETH Zurich
 Institute for Dynamic Systems and Control
 --
"""

import numpy as np

# Additional imports
import itertools
import Constants
import cProfile

# profiler = cProfile.Profile()

def solution(P, G, alpha):
    """Computes the optimal cost and the optimal control input for each 
    state of the state space solving the discounted stochastic shortest
    path problem by:
            - Value Iteration;
            - Policy Iteration;
            - Linear Programming; 
            - or a combination of these.

    Args:
        P  (np.array): A (K x K x L)-matrix containing the transition probabilities
                       between all states in the state space for all control inputs.
                       The entry P(i, j, l) represents the transition probability
                       from state i to state j if control input l is applied
        G  (np.array): A (K x L)-matrix containing the stage costs of all states in
                       the state space for all control inputs. The entry G(i, l)
                       represents the cost if we are in state i and apply control
                       input l
        alpha (float): The discount factor for the problem

    Returns:
        np.array: The optimal cost to go for the discounted stochastic SPP
        np.array: The optimal control policy for the discounted stochastic SPP

    """
    K = G.shape[0]

    # J_opt = np.zeros(K)
    J_opt = np.full(K, 1e03)    # Based on testing performance 
    u_opt = np.zeros(K) 
    
    # TODO implement Value Iteration, Policy Iteration, 
    #      Linear Programming or a combination of these
    t = np.arange(0, Constants.Constants.T)  
    z = np.arange(0, Constants.Constants.D)  
    y = np.arange(0, Constants.Constants.N)  
    x = np.arange(0, Constants.Constants.M)  
    state_space = np.array(list(itertools.product(t, z, y, x)))

    # Implementing memoization, keeps copy of next states and action for given state
    state_info_dict = {}

    delta_v = float('inf')
    epsilon = 9e-05

    while delta_v > epsilon:
        delta_v = 0

        # Keeping a copy so algorithm references previous values while this maintains current ones
        J_copy = np.copy(J_opt)

        for i in range(K):

            next_states_list = []
            possible_actions_list = []

            if i not in state_info_dict: 
                next_states_list = generate_possible_next_states(state_space[i], state_space)
                possible_actions_list = generate_possible_actions(state_space[i])

                state_info_dict[i] = [next_states_list, possible_actions_list]

            next_states_list = state_info_dict[i][0]
            possible_actions_list = state_info_dict[i][1]

            for action in possible_actions_list:
                action_total = 0

                for j in next_states_list:
                    transition_probability = P[i, j, action]
                    action_total += transition_probability*J_opt[j]

                stage_cost = G[i, action]
                value_action = stage_cost + alpha * action_total

                if value_action < J_copy[i]:
                    J_copy[i] = value_action
                    u_opt[i] = action
                
        delta_v = np.max(np.abs(J_copy - J_opt))
        J_opt = J_copy

    return J_opt, u_opt

# def solution(P, G, alpha):
#     K = G.shape[0]
#     L=3

#     # J_opt = np.zeros(K)
#     J_opt = np.full(K, 1e03)    # Based on testing performance 
#     u_opt = np.zeros(K) 
#     t = np.arange(0, Constants.Constants.T)  
#     z = np.arange(0, Constants.Constants.D)  
#     y = np.arange(0, Constants.Constants.N)  
#     x = np.arange(0, Constants.Constants.M)  
      
#     state_space = np.array(list(itertools.product(t, z, y, x)))

#     # Convergence parameters
#     epsilon = 1e-05
#     delta_v = float('inf')

#     while delta_v > epsilon:
#         J_copy = np.copy(J_opt)

#         # Vectorized computation for each action
#         for action in range(L):
#             # Compute the total cost for this action across all states
#             total_cost_action = G[:, action] + alpha * np.sum(P[:, :, action] * J_opt, axis=1)

#             # Update the optimal cost and policy
#             better_cost = total_cost_action < J_copy
#             J_copy[better_cost] = total_cost_action[better_cost]
#             u_opt[better_cost] = action

#         # Check for convergence
#         delta_v = np.max(np.abs(J_copy - J_opt))
#         J_opt = J_copy

#     return J_opt, u_opt


def freestyle_solution(Constants):
    """Computes the optimal cost and the optimal control input for each 
    state of the state space solving the discounted stochastic shortest
    path problem with a 200 MiB memory cap.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: The optimal cost to go for the discounted stochastic SPP
        np.array: The optimal control policy for the discounted stochastic SPP
    """
    K = Constants.T * Constants.D * Constants.N * Constants.M

    J_opt = np.zeros(K)
    u_opt = np.zeros(K)
    
    # TODO implement a solution that not necessarily adheres to
    #      the solution template. You are free to use
    #      compute_transition_probabilities and
    #      compute_stage_cost, but you are also free to introduce
    #      optimizations.

    return J_opt, u_opt

def generate_possible_next_states(current_state, state_space):
    '''
    Returns the index of the next possible states in the flattened array 
    '''
    possible_next_states = []

    t_i, z_i, y_i, x_i = current_state[0], current_state[1], current_state[2], current_state[3]

    if(t_i<(Constants.Constants.T-1)):
        t_j=t_i+1
    else:
        t_j=0
    
    if(z_i<(Constants.Constants.D-1)):
        z_up_j=z_i+1
    else:
        z_up_j=z_i
    
    if(z_i>0):
        z_down_j=z_i-1
    else:
        z_down_j=0

    if(y_i>0):
        y_north_j= y_i-1
    else:
        y_north_j=0
    
    if(y_i<(Constants.Constants.N-1)):
        y_south_j=y_i+1
    else:
        y_south_j= y_i
    
    if(x_i<(Constants.Constants.M-1)):
        x_east_j=x_i+1
    else:
        x_east_j=0

    if(x_i>0):
        x_west_j=x_i-1
    else:
        x_west_j=Constants.Constants.M-1

    j_up = map_state_to_index((t_j, z_up_j, y_i, x_i))
    j_stay=map_state_to_index((t_j, z_i, y_i, x_i))
    j_down=map_state_to_index((t_j, z_down_j, y_i, x_i))

    j_up_east=map_state_to_index((t_j, z_up_j, y_i, x_east_j))
    j_up_west=map_state_to_index((t_j, z_up_j, y_i, x_west_j))

    j_stay_east=map_state_to_index((t_j, z_i, y_i, x_east_j))
    j_stay_west=map_state_to_index((t_j, z_i, y_i, x_west_j))

    j_down_east=map_state_to_index((t_j, z_down_j, y_i, x_east_j))
    j_down_west=map_state_to_index((t_j, z_down_j, y_i, x_west_j))

    j_up_north=map_state_to_index((t_j, z_up_j, y_north_j, x_i))
    j_up_south=map_state_to_index((t_j, z_up_j, y_south_j, x_i))

    j_stay_north=map_state_to_index((t_j, z_i, y_north_j, x_i))
    j_stay_south=map_state_to_index((t_j, z_i, y_south_j, x_i))

    j_down_north=map_state_to_index((t_j, z_down_j, y_north_j, x_i))
    j_down_south=map_state_to_index((t_j, z_down_j, y_south_j, x_i))

    if(z_i<(Constants.Constants.D-1) and z_i > 0):
        possible_next_states = [
            j_up,
            j_stay,
            j_down,

            j_up_east,
            j_up_west,

            j_stay_east,
            j_stay_west,

            j_down_east,
            j_down_west,

            j_up_north,
            j_up_south,

            j_stay_north,
            j_stay_south,

            j_down_north,
            j_down_south
        ]
    elif z_i == (Constants.Constants.D-1):
        possible_next_states = [
            j_stay,
            j_down,

            j_stay_east,
            j_stay_west,

            j_down_east,
            j_down_west,
            
            j_stay_north,
            j_stay_south,

            j_down_north,
            j_down_south
        ]
    elif z_i == 0:
        possible_next_states = [
            j_up,
            j_stay,

            j_up_east,
            j_up_west,

            j_stay_east,
            j_stay_west,
            
            j_up_north,
            j_up_south,

            j_stay_north,
            j_stay_south
        ]

    return list(set(possible_next_states))

def map_state_to_index(input_state):
    '''
    Maps a state to the index in the P matrix 
    '''
    t_in, z_in, y_in, x_in = input_state[0], input_state[1], input_state[2], input_state[3]

    return t_in*(Constants.Constants.D*Constants.Constants.N*Constants.Constants.M) + z_in*(Constants.Constants.N*Constants.Constants.M) + y_in*Constants.Constants.M + x_in

def generate_possible_actions(current_state):
    '''
    Returns a list of possible actions given a current state. 
    '''
    z_i = current_state[1]

    if(z_i<(Constants.Constants.D-1) and z_i > 0):
        return [Constants.Constants.V_DOWN, Constants.Constants.V_STAY, Constants.Constants.V_UP]
    elif z_i == (Constants.Constants.D-1):
        return [Constants.Constants.V_DOWN, Constants.Constants.V_STAY]
    elif z_i == 0:
        return [Constants.Constants.V_STAY, Constants.Constants.V_UP]

    print("Error with generate_possible_actions!!!")
    return []