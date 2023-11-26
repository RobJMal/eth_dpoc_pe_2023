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
    # TODO fill the transition probability matrix P here
    for i in range (K):#K
        # i = np.where((state_space == (4 ,3,4, 4)).all(axis=1))[0][0]
        t_i=state_space[i][0]
        z_i=state_space[i][1]
        y_i=state_space[i][2]
        x_i=state_space[i][3]
        
        print_state("current state: ", i, state_space)

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

        # if(y_i>0):
        #     y_north_j= y_i-1
        # else:
        #     y_north_j=0
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
        
        
        j_up=np.where((state_space == (t_j, z_up_j, y_i, x_i)).all(axis=1))[0][0]
        j_stay=np.where((state_space == (t_j, z_i, y_i, x_i)).all(axis=1))[0][0]
        j_down=np.where((state_space == (t_j, z_down_j, y_i, x_i)).all(axis=1))[0][0]

        j_up_east=np.where((state_space == (t_j, z_up_j, y_i, x_east_j)).all(axis=1))[0][0]
        j_up_west=np.where((state_space == (t_j, z_up_j, y_i, x_west_j)).all(axis=1))[0][0]

        j_stay_east=np.where((state_space == (t_j, z_i, y_i, x_east_j)).all(axis=1))[0][0]
        j_stay_west=np.where((state_space == (t_j, z_i, y_i, x_west_j)).all(axis=1))[0][0]

        j_down_east=np.where((state_space == (t_j, z_down_j, y_i, x_east_j)).all(axis=1))[0][0]
        j_down_west=np.where((state_space == (t_j, z_down_j, y_i, x_west_j)).all(axis=1))[0][0]

        j_up_north=np.where((state_space == (t_j, z_up_j, y_north_j, x_i)).all(axis=1))[0][0]
        j_up_south=np.where((state_space == (t_j, z_up_j, y_south_j, x_i)).all(axis=1))[0][0]

        j_stay_north=np.where((state_space == (t_j, z_i, y_north_j, x_i)).all(axis=1))[0][0]
        j_stay_south=np.where((state_space == (t_j, z_i, y_south_j, x_i)).all(axis=1))[0][0]

        j_down_north=np.where((state_space == (t_j, z_down_j, y_north_j, x_i)).all(axis=1))[0][0]
        J_down_south=np.where((state_space == (t_j, z_down_j, y_south_j, x_i)).all(axis=1))[0][0]

        # print("Next states -----")
        # print_state("up: ", j_up, state_space)
        # print_state("stay: ", j_stay, state_space)
        # print_state("down: ", j_down, state_space)

        # print_state("up east: ", j_up_east, state_space)
        # print_state("up west: ", j_up_west, state_space)

        # print_state("stay east: ", j_stay_east, state_space)
        # print_state("stay west: ", j_stay_west, state_space)

        # print_state("down east: ", j_down_east, state_space)
        # print_state("down west: ", j_down_west, state_space)

        # print_state("up north: ", j_up_north, state_space)
        # print_state("up south: ", j_up_south, state_space)

        # print_state("stay north: ", j_stay_north, state_space)
        # print_state("stay south: ", j_stay_south, state_space)

        # print_state("down north: ", j_down_north, state_space)
        # print_state("down south: ", J_down_south, state_space)
        # print("------------")
        # print()

        y_north_limit = 0
        y_south_limit = 0
        # if y_i == 0:
        #     y_north_limit = 1
        # if y_i == Constants.N - 1:
        #     y_south_limit = 1
        if y_i == Constants.N - 1:
            y_north_limit = 1
        if y_i == 0:
            y_south_limit = 1

        # Constants.V_DOWN
        if z_i == 0: 
            P[i,j_up,Constants.V_DOWN]=0
            P[i,j_stay,Constants.V_DOWN]=0
            P[i,j_down,Constants.V_DOWN]=0

            P[i,j_up_east,Constants.V_DOWN]=0
            P[i,j_up_west,Constants.V_DOWN]=0

            P[i,j_stay_east,Constants.V_DOWN]=0
            P[i,j_stay_west,Constants.V_DOWN]=0

            P[i,j_down_east,Constants.V_DOWN]=0
            P[i,j_down_west,Constants.V_DOWN]=0

            P[i,j_up_north,Constants.V_DOWN]=0
            P[i,j_up_south,Constants.V_DOWN]=0

            P[i, j_stay_north,Constants.V_DOWN]=0
            P[i,j_stay_south,Constants.V_DOWN]=0

            P[i,j_down_north,Constants.V_DOWN]=0
            P[i,J_down_south,Constants.V_DOWN]=0
        else: 
            P[i,j_up,Constants.V_DOWN]=0
            P[i,j_stay,Constants.V_DOWN]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]
            P[i,j_down,Constants.V_DOWN]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_STAY]

            P[i,j_up_east,Constants.V_DOWN]=0
            P[i,j_up_west,Constants.V_DOWN]=0

            P[i,j_stay_east,Constants.V_DOWN]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
            P[i,j_stay_west,Constants.V_DOWN]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]

            P[i,j_down_east,Constants.V_DOWN]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
            P[i,j_down_west,Constants.V_DOWN]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]##########

            P[i,j_up_north,Constants.V_DOWN]=0
            P[i,j_up_south,Constants.V_DOWN]=0

            P[i, j_stay_north,Constants.V_DOWN]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
            P[i,j_stay_south,Constants.V_DOWN]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]

            P[i,j_down_north,Constants.V_DOWN]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
            P[i,J_down_south,Constants.V_DOWN]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]

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

        # Constants.V_STAY (always possible)
        
        P[i,j_down_north,Constants.V_STAY]=0
        P[i,J_down_south,Constants.V_STAY]=0
        P[i,j_up,Constants.V_STAY]=0
        P[i,j_up_north,Constants.V_STAY]=0
        P[i,j_up_south,Constants.V_STAY]=0
        P[i,j_down_east,Constants.V_STAY]=0
        P[i,j_down_west,Constants.V_STAY]=0
        P[i,j_down,Constants.V_STAY]=0

        P[i,j_up_east,Constants.V_STAY]=0
        P[i,j_up_west,Constants.V_STAY]=0

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
            
        # if z_i == 0:
        #     P[i,j_stay_east,Constants.V_STAY]=Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_EAST]
        #     P[i,j_stay_west,Constants.V_STAY]=Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_WEST]
        #     P[i, j_stay_north,Constants.V_STAY]=Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_NORTH]
        #     P[i,j_stay_south,Constants.V_STAY]=Constants.P_H_TRANSITION[z_i].P_WIND[Constants.H_SOUTH]

            
               

        # Constants.V_UP
        if (z_i == (Constants.D - 1)):
            P[i,j_up,Constants.V_UP]=0
            P[i,j_stay,Constants.V_UP]=0
            P[i,j_down,Constants.V_UP]=0

            P[i,j_up_east,Constants.V_UP]=0
            P[i,j_up_west,Constants.V_UP]=0

            P[i,j_stay_east,Constants.V_UP]=0
            P[i,j_stay_west,Constants.V_UP]=0

            P[i,j_down_east,Constants.V_UP]=0
            P[i,j_down_west,Constants.V_UP]=0

            P[i,j_up_north,Constants.V_UP]=0
            P[i,j_up_south,Constants.V_UP]=0

            P[i, j_stay_north,Constants.V_UP]=0
            P[i,j_stay_south,Constants.V_UP]=0

            P[i,j_down_north,Constants.V_UP]=0
            P[i,J_down_south,Constants.V_UP]=0
        else:
            P[i,j_down_east,Constants.V_UP]=0
            P[i,j_down_west,Constants.V_UP]=0
            P[i,j_down,Constants.V_UP]=0
            P[i,j_down_north,Constants.V_UP]=0
            
            P[i,J_down_south,Constants.V_UP]=0

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
    
    print_probabilities("State space i", i, state_space, Constants, P)
    # print(P.shape)
    # non_zero_indices = np.nonzero(P)
    # for index in zip(*non_zero_indices):
    #     print(f"P{index} = {P[index]}")

    return P

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