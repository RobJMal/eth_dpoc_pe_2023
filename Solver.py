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
    from ComputeStageCosts import compute_stage_cost

    def compute_transition_probabilities_sparse(Constants):
        """Computes the transition probability matrix P.

        It is of size (K,K,L) where:
            - K is the size of the state space;
            - L is the size of the input space; and
            - P[i,j,l] corresponds to the probability of transitioning
                from the state i to the state j when input l is applied.

        Args:
            Constants: The constants describing the problem instance.

        Returns:
            scipy.sparse: Transition probability matrix of shape 
        """
        import itertools
        from scipy.sparse import coo_matrix

        

        t = np.arange(0, Constants.T)  
        z = np.arange(0, Constants.D)  
        y = np.arange(0, Constants.N)  
        x = np.arange(0, Constants.M)  
        state_space = np.array(list(itertools.product(t, z, y, x)))

        K = Constants.T * Constants.D * Constants.N * Constants.M
        input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
        L = len(input_space)

        # Initialize lists for COO format
        rows = []
        cols = []
        data = []
        matrix_dict={}

        def add_to_sparse_matrix(i, j, l, value):
            """
            Adds a non-zero transition probability to the sparse matrix.
            Adding to COO matrix. 

            Args:
                - matrix_dict (dict): Dictionary for helping construct the coo matrix 
                - i (int): Current state index.
                - j (int): Next state index.
                - l (int): Action index.
                - value (float): Probability value to be added.
                - L (int): Total number of actions.
            """
            if value != 0.0:
                row_index= i*L+l
                matrix_dict[(row_index,j)]= value

        for i in range (K):
            t_i=state_space[i][0]
            z_i=state_space[i][1]
            y_i=state_space[i][2]
            x_i=state_space[i][3]

            if(t_i<(Constants.T-1)):
                t_j=t_i+1
            else:
                t_j=0
            
            if(z_i<(Constants.D-1)):
                z_up_j=z_i+1
            else:
                z_up_j=z_i
            
            if(z_i>0):
                z_down_j=z_i-1
            else:
                z_down_j=0

            if(y_i<(Constants.N-1)):
                y_north_j= y_i+1
            else:
                y_north_j=y_i
            
            if(y_i>0):
                y_south_j=y_i-1
            else:
                y_south_j= 0
            
            if(x_i<(Constants.M-1)):
                x_east_j=x_i+1
            else:
                x_east_j=0

            if(x_i>0):
                x_west_j=x_i-1
            else:
                x_west_j=Constants.M-1
            
            j_up = t_j*(Constants.D*Constants.N*Constants.M) + z_up_j*(Constants.N*Constants.M) + y_i*Constants.M + x_i
            j_stay = t_j*(Constants.D*Constants.N*Constants.M) + z_i*(Constants.N*Constants.M) + y_i*Constants.M + x_i
            j_down = t_j*(Constants.D*Constants.N*Constants.M) + z_down_j*(Constants.N*Constants.M) + y_i*Constants.M + x_i

            j_up_east = t_j*(Constants.D*Constants.N*Constants.M) + z_up_j*(Constants.N*Constants.M) + y_i*Constants.M + x_east_j
            j_up_west = t_j*(Constants.D*Constants.N*Constants.M) + z_up_j*(Constants.N*Constants.M) + y_i*Constants.M + x_west_j

            j_stay_east = t_j*(Constants.D*Constants.N*Constants.M) + z_i*(Constants.N*Constants.M) + y_i*Constants.M + x_east_j
            j_stay_west = t_j*(Constants.D*Constants.N*Constants.M) + z_i*(Constants.N*Constants.M) + y_i*Constants.M + x_west_j

            j_down_east = t_j*(Constants.D*Constants.N*Constants.M) + z_down_j*(Constants.N*Constants.M) + y_i*Constants.M + x_east_j
            j_down_west = t_j*(Constants.D*Constants.N*Constants.M) + z_down_j*(Constants.N*Constants.M) + y_i*Constants.M + x_west_j

            j_up_north = t_j*(Constants.D*Constants.N*Constants.M) + z_up_j*(Constants.N*Constants.M) + y_north_j*Constants.M + x_i
            j_up_south = t_j*(Constants.D*Constants.N*Constants.M) + z_up_j*(Constants.N*Constants.M) + y_south_j*Constants.M + x_i

            j_stay_north = t_j*(Constants.D*Constants.N*Constants.M) + z_i*(Constants.N*Constants.M) + y_north_j*Constants.M + x_i
            j_stay_south = t_j*(Constants.D*Constants.N*Constants.M) + z_i*(Constants.N*Constants.M) + y_south_j*Constants.M + x_i

            j_down_north = t_j*(Constants.D*Constants.N*Constants.M) + z_down_j*(Constants.N*Constants.M) + y_north_j*Constants.M + x_i
            j_down_south = t_j*(Constants.D*Constants.N*Constants.M) + z_down_j*(Constants.N*Constants.M) + y_south_j*Constants.M + x_i

            y_north_limit = 0
            y_south_limit = 0

            if y_i == Constants.N - 1:
                y_north_limit = 1
            if y_i == 0:
                y_south_limit = 1

            # ----- Constants.V_DOWN -----
            if z_i > 0: 
                p_value_j_stay = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
                p_value_j_down = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
                add_to_sparse_matrix(i, j_stay, Constants.V_DOWN, p_value_j_stay)
                add_to_sparse_matrix(i, j_down, Constants.V_DOWN, p_value_j_down)

                p_value_j_stay_east = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
                p_value_j_stay_west = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]
                add_to_sparse_matrix(i, j_stay_east, Constants.V_DOWN, p_value_j_stay_east)
                add_to_sparse_matrix(i, j_stay_west, Constants.V_DOWN, p_value_j_stay_west)

                p_value_j_down_east = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
                p_value_j_down_west = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]
                add_to_sparse_matrix(i, j_down_east, Constants.V_DOWN, p_value_j_down_east)
                add_to_sparse_matrix(i, j_down_west, Constants.V_DOWN, p_value_j_down_west)

                p_value_j_stay_north = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
                p_value_j_stay_south = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]
                add_to_sparse_matrix(i, j_stay_north, Constants.V_DOWN, p_value_j_stay_north)
                add_to_sparse_matrix(i, j_stay_south, Constants.V_DOWN, p_value_j_stay_south)

                p_value_j_down_north = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
                p_value_j_down_south = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]
                add_to_sparse_matrix(i, j_down_north, Constants.V_DOWN, p_value_j_down_north)
                add_to_sparse_matrix(i, j_down_south, Constants.V_DOWN, p_value_j_down_south)

                if y_north_limit:   # If north limit, akin to staying 
                    p_value_j_stay = (Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                        Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
                    p_value_j_down = (Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]+
                                        Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
                    add_to_sparse_matrix(i, j_stay, Constants.V_DOWN, p_value_j_stay)
                    add_to_sparse_matrix(i, j_down, Constants.V_DOWN, p_value_j_down)

                elif y_south_limit:   # If north limit, akin to staying 
                    p_value_j_stay = (Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                        Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])
                    p_value_j_down = (Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]+
                                        Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])
                    add_to_sparse_matrix(i, j_stay, Constants.V_DOWN, p_value_j_stay)
                    add_to_sparse_matrix(i, j_down, Constants.V_DOWN, p_value_j_down)

            # ----- Constants.V_STAY (always possible) -----
            p_value_j_stay = Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
            add_to_sparse_matrix(i, j_stay, Constants.V_STAY, p_value_j_stay)
        
            p_value_j_stay_east = Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
            p_value_j_stay_west = Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]
            add_to_sparse_matrix(i, j_stay_east, Constants.V_STAY, p_value_j_stay_east)
            add_to_sparse_matrix(i, j_stay_west, Constants.V_STAY, p_value_j_stay_west)

            p_value_j_stay_north = Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
            p_value_j_stay_south = Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]
            add_to_sparse_matrix(i, j_stay_north, Constants.V_STAY, p_value_j_stay_north)
            add_to_sparse_matrix(i, j_stay_south, Constants.V_STAY, p_value_j_stay_south)

            if y_north_limit:   # If north limit, akin to staying 
                p_value_j_stay = (Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                    Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
                add_to_sparse_matrix(i, j_stay, Constants.V_STAY, p_value_j_stay)

            elif y_south_limit:   # If north limit, akin to staying 
                p_value_j_stay = (Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                    Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])
                add_to_sparse_matrix(i, j_stay, Constants.V_STAY, p_value_j_stay)

            # ----- Constants.V_UP -----
            if (z_i < (Constants.D - 1)):
                p_value_j_up = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
                p_value_j_stay = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
                add_to_sparse_matrix(i, j_up, Constants.V_UP, p_value_j_up)
                add_to_sparse_matrix(i, j_stay, Constants.V_UP, p_value_j_stay)
                
                p_value_j_up_east = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
                p_value_j_up_west = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]
                add_to_sparse_matrix(i, j_up_east, Constants.V_UP, p_value_j_up_east)
                add_to_sparse_matrix(i, j_up_west, Constants.V_UP, p_value_j_up_west)

                p_value_j_stay_east = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
                p_value_j_stay_west = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]
                add_to_sparse_matrix(i, j_stay_east, Constants.V_UP, p_value_j_stay_east)
                add_to_sparse_matrix(i, j_stay_west, Constants.V_UP, p_value_j_stay_west)

                p_value_j_up_north = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
                p_value_j_up_south = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]
                add_to_sparse_matrix(i, j_up_north, Constants.V_UP, p_value_j_up_north)
                add_to_sparse_matrix(i, j_up_south, Constants.V_UP, p_value_j_up_south)

                p_value_j_stay_north = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
                p_value_j_stay_south = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]
                add_to_sparse_matrix(i, j_stay_north, Constants.V_UP, p_value_j_stay_north)
                add_to_sparse_matrix(i, j_stay_south, Constants.V_UP, p_value_j_stay_south)

                if y_north_limit:   # If north limit, akin to staying 
                    p_value_j_stay = (Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                        Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
                    p_value_j_up = (Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]+
                                        Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
                    add_to_sparse_matrix(i, j_stay, Constants.V_UP, p_value_j_stay)
                    add_to_sparse_matrix(i, j_up, Constants.V_UP, p_value_j_up)
                    
                elif y_south_limit:   # If north limit, akin to staying 
                    p_value_j_stay = (Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                        Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])
                    p_value_j_up = (Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]+
                                        Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])
                    add_to_sparse_matrix(i, j_stay, Constants.V_UP, p_value_j_stay)
                    add_to_sparse_matrix(i, j_up, Constants.V_UP, p_value_j_up)

        rows, cols, data = zip(*[(key[0], key[1], val) for key, val in matrix_dict.items()])
        P_sparse = coo_matrix((data, (rows, cols)), shape=(K*L, K))

        return P_sparse

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
