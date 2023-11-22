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

    J_opt = np.zeros(K)
    u_opt = np.zeros(K) 
    
    # TODO implement Value Iteration, Policy Iteration, 
    #      Linear Programming or a combination of these
    delta_v = float('inf')
    epsilon = 0.001

    while delta_v > epsilon:
        delta_v = 0

        # Keeping a copy so algorithm references previous values while this maintains current ones
        J_copy = np.zeros(K)

        for current_state in range(K):

            next_states_list = generate_possible_next_states(current_state)
            possible_actions_list = generate_possible_actions(current_state)

                for action in possible_actions_list:
                    action_total = 0

                    for next_state in next_states_list:
                        transition_probability = P[current_state, next_state, action]
                        reward = G[current_state, action]
                        action_total += transition_probability*(reward + alpha * J_opt[next_state])

                    if action_total > J_copy[current_state]:
                        J_copy[current_state] = action_total
                        u_opt[current_state] = action

        delta_v = np.max(np.abs(J_copy - J_opt))
        J_opt = J_copy

    return J_opt, u_opt

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

def generate_possible_next_states(current_state):
    '''
    Returns a list of possible next states given a current state. 
    '''

    return []

def generate_possible_actions(current_state):
    '''
    Returns a list of possible actions given a current state. 
    '''

    return []