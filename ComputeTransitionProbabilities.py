"""
 ComputeTransitionProbabilities.py

 Python function template to compute the transition probability matrix.

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
from scipy.sparse import coo_matrix

def compute_transition_probabilities(Constants):
    """Computes the transition probability matrix P.

    It is of size (K,K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - P[i,j,l] corresponds to the probability of transitioning
            from the state i to the state j when input l is applied.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Transition probability matrix of shape (K,K,L).
    """
    t = np.arange(0, Constants.T)  
    z = np.arange(0, Constants.D)  
    y = np.arange(0, Constants.N)  
    x = np.arange(0, Constants.M)  
    state_space = np.array(list(itertools.product(t, z, y, x)))

    K = Constants.T * Constants.D * Constants.N * Constants.M
    input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
    L = len(input_space)

    P = np.zeros((K, K, L))
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
        
        j_up=map_state_to_index((t_j, z_up_j, y_i, x_i))
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

        y_north_limit = 0
        y_south_limit = 0

        if y_i == Constants.N - 1:
            y_north_limit = 1
        if y_i == 0:
            y_south_limit = 1

        # ----- Constants.V_DOWN -----
        if z_i > 0: 
            P[i,j_stay,Constants.V_DOWN]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
            P[i,j_down,Constants.V_DOWN]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]

            P[i,j_stay_east,Constants.V_DOWN]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
            P[i,j_stay_west,Constants.V_DOWN]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]

            P[i,j_down_east,Constants.V_DOWN]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
            P[i,j_down_west,Constants.V_DOWN]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]

            P[i, j_stay_north,Constants.V_DOWN]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
            P[i,j_stay_south,Constants.V_DOWN]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]

            P[i,j_down_north,Constants.V_DOWN]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
            P[i,j_down_south,Constants.V_DOWN]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]

            if y_north_limit:   # If north limit, akin to staying 
                P[i,j_stay,Constants.V_DOWN]=(Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                              Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
                P[i,j_down,Constants.V_DOWN]=(Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]+
                                              Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])

            elif y_south_limit:   # If north limit, akin to staying 
                P[i,j_stay,Constants.V_DOWN]=(Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                              Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])
                P[i,j_down,Constants.V_DOWN]=(Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]+
                                              Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])

        # ----- Constants.V_STAY (always possible) -----
        P[i,j_stay,Constants.V_STAY]=Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
       
        P[i,j_stay_east,Constants.V_STAY]=Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
        P[i,j_stay_west,Constants.V_STAY]=Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]

        P[i, j_stay_north,Constants.V_STAY]=Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
        P[i,j_stay_south,Constants.V_STAY]=Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]

        if y_north_limit:   # If north limit, akin to staying 
            P[i,j_stay,Constants.V_STAY]=(Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                              Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])

        elif y_south_limit:   # If north limit, akin to staying 
            P[i,j_stay,Constants.V_STAY]=(Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                            Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])

        # ----- Constants.V_UP -----
        if (z_i < (Constants.D - 1)):
            P[i,j_up,Constants.V_UP]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
            P[i,j_stay,Constants.V_UP]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
            
            P[i,j_up_east,Constants.V_UP]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
            P[i,j_up_west,Constants.V_UP]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]

            P[i,j_stay_east,Constants.V_UP]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
            P[i,j_stay_west,Constants.V_UP]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]

            P[i,j_up_north,Constants.V_UP]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
            P[i,j_up_south,Constants.V_UP]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]

            P[i, j_stay_north,Constants.V_UP]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
            P[i,j_stay_south,Constants.V_UP]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]

            if y_north_limit:   # If north limit, akin to staying 
                P[i,j_stay,Constants.V_UP]=(Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                              Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
                P[i,j_up,Constants.V_UP]=(Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]+
                                              Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
                
            elif y_south_limit:   # If north limit, akin to staying 
                P[i,j_stay,Constants.V_UP]=(Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                              Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])
                P[i,j_up,Constants.V_UP]=(Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]+
                                              Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])

    return P

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
        
        j_up=map_state_to_index((t_j, z_up_j, y_i, x_i))
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

        y_north_limit = 0
        y_south_limit = 0

        if y_i == Constants.N - 1:
            y_north_limit = 1
        if y_i == 0:
            y_south_limit = 1

        # Constants.V_DOWN
        if z_i == 0: 
            dummy_var = 0 
        else: 
            p_value_j_stay = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
            p_value_j_down = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
            add_to_sparse_matrix(rows, cols, data, i, j_stay, Constants.V_DOWN, p_value_j_stay, L)
            add_to_sparse_matrix(rows, cols, data, i, j_down, Constants.V_DOWN, p_value_j_down, L)

            p_value_j_stay_east = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
            p_value_j_stay_west = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]
            add_to_sparse_matrix(rows, cols, data, i, j_stay_east, Constants.V_DOWN, p_value_j_stay_east, L)
            add_to_sparse_matrix(rows, cols, data, i, j_stay_west, Constants.V_DOWN, p_value_j_stay_west, L)

            p_value_j_down_east = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
            p_value_j_down_west = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]
            add_to_sparse_matrix(rows, cols, data, i, j_down_east, Constants.V_DOWN, p_value_j_down_east, L)
            add_to_sparse_matrix(rows, cols, data, i, j_down_west, Constants.V_DOWN, p_value_j_down_west, L)

            p_value_j_stay_north = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
            p_value_j_stay_south = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]
            add_to_sparse_matrix(rows, cols, data, i, j_stay_north, Constants.V_DOWN, p_value_j_stay_north, L)
            add_to_sparse_matrix(rows, cols, data, i, j_stay_south, Constants.V_DOWN, p_value_j_stay_south, L)

            p_value_j_down_north = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
            p_value_j_down_south = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]
            add_to_sparse_matrix(rows, cols, data, i, j_down_north, Constants.V_DOWN, p_value_j_down_north, L)
            add_to_sparse_matrix(rows, cols, data, i, j_down_south, Constants.V_DOWN, p_value_j_down_south, L)


            if y_north_limit:   # If north limit, akin to staying 
                p_value_j_stay = (Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                    Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
                p_value_j_down = (Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]+
                                    Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
                add_to_sparse_matrix(rows, cols, data, i, j_stay, Constants.V_DOWN, p_value_j_stay, L)
                add_to_sparse_matrix(rows, cols, data, i, j_down, Constants.V_DOWN, p_value_j_down, L)

            elif y_south_limit:   # If north limit, akin to staying 
                p_value_j_stay = (Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                    Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])
                p_value_j_down = (Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]+
                                    Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])
                add_to_sparse_matrix(rows, cols, data, i, j_stay, Constants.V_DOWN, p_value_j_stay, L)
                add_to_sparse_matrix(rows, cols, data, i, j_down, Constants.V_DOWN, p_value_j_down, L)

        # ----- Constants.V_STAY (always possible) -----
        p_value_j_stay = Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
        add_to_sparse_matrix(rows, cols, data, i, j_stay, Constants.V_STAY, p_value_j_stay, L)
       
        p_value_j_stay_east = Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
        p_value_j_stay_west = Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]
        add_to_sparse_matrix(rows, cols, data, i, j_stay_east, Constants.V_STAY, p_value_j_stay_east, L)
        add_to_sparse_matrix(rows, cols, data, i, j_stay_west, Constants.V_STAY, p_value_j_stay_west, L)

        p_value_j_stay_north = Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
        p_value_j_stay_south = Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]
        add_to_sparse_matrix(rows, cols, data, i, j_stay_north, Constants.V_STAY, p_value_j_stay_north, L)
        add_to_sparse_matrix(rows, cols, data, i, j_stay_south, Constants.V_STAY, p_value_j_stay_south, L)

        if y_north_limit:   # If north limit, akin to staying 
            p_value_j_stay = (Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
            add_to_sparse_matrix(rows, cols, data, i, j_stay, Constants.V_STAY, p_value_j_stay, L)

        elif y_south_limit:   # If north limit, akin to staying 
            p_value_j_stay = (Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])
            add_to_sparse_matrix(rows, cols, data, i, j_stay, Constants.V_STAY, p_value_j_stay, L)

        # ----- Constants.V_UP -----
        if (z_i == (Constants.D - 1)):
            dummy_var = 1
        else:
            p_value_j_up = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
            p_value_j_stay = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
            add_to_sparse_matrix(rows, cols, data, i, j_up, Constants.V_UP, p_value_j_up, L)
            add_to_sparse_matrix(rows, cols, data, i, j_stay, Constants.V_UP, p_value_j_stay, L)
            
            p_value_j_up_east = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
            p_value_j_up_west = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]
            add_to_sparse_matrix(rows, cols, data, i, j_up_east, Constants.V_UP, p_value_j_up_east, L)
            add_to_sparse_matrix(rows, cols, data, i, j_up_west, Constants.V_UP, p_value_j_up_west, L)

            p_value_j_stay_east = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
            p_value_j_stay_west = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]
            add_to_sparse_matrix(rows, cols, data, i, j_stay_east, Constants.V_UP, p_value_j_stay_east, L)
            add_to_sparse_matrix(rows, cols, data, i, j_stay_west, Constants.V_UP, p_value_j_stay_west, L)

            p_value_j_up_north = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
            p_value_j_up_south = Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]
            add_to_sparse_matrix(rows, cols, data, i, j_up_north, Constants.V_UP, p_value_j_up_north, L)
            add_to_sparse_matrix(rows, cols, data, i, j_up_south, Constants.V_UP, p_value_j_up_south, L)

            p_value_j_stay_north = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
            p_value_j_stay_south = Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]
            add_to_sparse_matrix(rows, cols, data, i, j_stay_north, Constants.V_UP, p_value_j_stay_north, L)
            add_to_sparse_matrix(rows, cols, data, i, j_stay_south, Constants.V_UP, p_value_j_stay_south, L)

            if y_north_limit:   # If north limit, akin to staying 
                p_value_j_stay = (Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                    Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
                p_value_j_up = (Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]+
                                    Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH])
                add_to_sparse_matrix(rows, cols, data, i, j_stay, Constants.V_UP, p_value_j_stay, L)
                add_to_sparse_matrix(rows, cols, data, i, j_up, Constants.V_UP, p_value_j_up, L)
                
            elif y_south_limit:   # If north limit, akin to staying 
                p_value_j_stay = (Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY] +
                                    Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])
                p_value_j_up = (Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]+
                                    Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH])
                add_to_sparse_matrix(rows, cols, data, i, j_stay, Constants.V_UP, p_value_j_stay, L)
                add_to_sparse_matrix(rows, cols, data, i, j_up, Constants.V_UP, p_value_j_up, L)

    P_sparse = coo_matrix((data, (rows, cols)), shape=(K*L, K))

    return P_sparse

def print_state(state_name, i, state_space):
    t_i=state_space[i][0]
    z_i=state_space[i][1]
    y_i=state_space[i][2]
    x_i=state_space[i][3]

    print(state_name + "(" + str(t_i) + ", " + str(z_i) + ", " + str(y_i) + ", " + str(x_i) + ")")

def print_probabilities(name, i, state_space, Constants, P):

    matrix_np=np.array(P)
    sum_matrix=np.sum(matrix_np, axis=1)
    
    print(name + ", V_UP: " + str(sum_matrix[i][Constants.V_UP]))
    print(name + ", V_STAY: " + str(sum_matrix[i][Constants.V_STAY]))
    print(name + ", V_DOWN: " + str(sum_matrix[i][Constants.V_DOWN]))

def map_state_to_index(input_state):
    '''
    Maps a state to the index in the P matrix 
    '''
    t_in, z_in, y_in, x_in = input_state[0], input_state[1], input_state[2], input_state[3]

    return t_in*(Constants.Constants.D*Constants.Constants.N*Constants.Constants.M) + z_in*(Constants.Constants.N*Constants.Constants.M) + y_in*Constants.Constants.M + x_in

def add_to_sparse_matrix(rows, cols, data, i, j, l, value, L):
    """
    Adds a non-zero transition probability to the sparse matrix.
    Adding to COO matrix. 

    Args:
    rows (list): List of row indices for the sparse matrix.
    cols (list): List of column indices for the sparse matrix.
    data (list): List of data values for the sparse matrix.
    i (int): Current state index.
    j (int): Next state index.
    l (int): Action index.
    value (float): Probability value to be added.
    L (int): Total number of actions.

    """
    if value != 0.0:
        row_index = i * L + l
        rows.append(row_index)
        cols.append(j)
        data.append(value)

def coo_to_3d(P_sparse, K, L):
    """
    Converts a COO sparse matrix to a 3D numpy array.

    Args:
    P_sparse (coo_matrix): The COO sparse matrix to be converted.
    K (int): Size of the state space.
    L (int): Size of the action space.

    Returns:
    np.array: A 3D numpy array with shape (K, K, L).
    """
    P_3d = np.zeros((K, K, L))

    for i, j, value in zip(P_sparse.row, P_sparse.col, P_sparse.data):
        state_index = i // L
        action_index = i % L
        P_3d[state_index, j, action_index] = value

    return P_3d
