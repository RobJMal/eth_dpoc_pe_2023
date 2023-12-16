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
    from scipy.sparse import csr_matrix
    K, L = G.shape

    P_csr = [csr_matrix(P[:, :, action]) for action in range(L)]

    J_opt = np.full(K, 1e03)   
    u_opt = np.zeros(K) 

    epsilon = 9e-05
    delta_v = float('inf')

    while delta_v > epsilon:
        J_prev = J_opt.copy()  

        for action in range(L):
            J_opt_col_vector = J_opt.reshape(-1, 1)
            
            total_cost_action = G[:, action] + alpha * (P_csr[action].dot(J_opt_col_vector)).flatten()


            better_cost = total_cost_action < J_opt
            J_opt[better_cost] = total_cost_action[better_cost]
            u_opt[better_cost] = action

        delta_v = np.max(np.abs(J_opt - J_prev))

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
    from ComputeTransitionProbabilities import compute_transition_probabilities_sparse
    from ComputeStageCosts import compute_stage_cost
    K = Constants.T * Constants.D * Constants.N * Constants.M

    J_opt = np.zeros(K)
    u_opt = np.zeros(K)
    P=compute_transition_probabilities_sparse(Constants)
    G=compute_stage_cost(Constants)

    K, L = G.shape
    P = P.tocsr()
    J_opt = np.full(K, 1e03)    # Based on testing performance 
    u_opt = np.zeros(K) 

    # Convergence parameters
    epsilon = 9e-05
    delta_v = float('inf')

    while delta_v > epsilon:
        J_prev = J_opt.copy() 

        for action in range(L):
            action_indices = np.arange(action, K * L, L)
            P_action = P[action_indices, :]
            total_cost_action = G[:, action] + Constants.ALPHA * P_action.dot(J_opt)

            better_cost = total_cost_action < J_opt
            J_opt[better_cost] = total_cost_action[better_cost]
            u_opt[better_cost] = action

        delta_v = np.max(np.abs(J_opt - J_prev))

    return J_opt, u_opt
