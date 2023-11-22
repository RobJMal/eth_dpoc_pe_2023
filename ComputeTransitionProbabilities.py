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
    import itertools

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
    for i in range (K):
        x=0
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

        if(y_i>0):
            y_north_j= y_i-1
        else:
            y_north_j=0
        
        if(y_i<(Constants.N-1)):
            y_south_j=y_i+1
        else:
            y_south_j= y_i
        
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


        if((z_i< Constants.D-1) and (z_i>0) and (y_i>0) and (y_i<(Constants.N-1)) and (x_i>0)and (x_i< Constants.M)):
            for l in range(L):
                match l:
                    case Constants.V_DOWN:

                        P[i,j_up,l]=0
                        P[i,j_stay,l]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[H_STAY]
                        P[i,j_down,l]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[H_STAY]

                        P[i,j_up_east,l]=0
                        P[i,j_up_west,l]=0

                        P[i,j_stay_east,l]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[H_EAST]
                        P[i,j_stay_west,l]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[H_WEST]

                        P[i,j_down_east,l]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[H_EAST]
                        P[i,j_down_west,l]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[H_WEST]

                        P[i,j_up_north,l]=0
                        P[i,j_up_south,l]=0

                        P[i, j_stay_north,l]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[H_NORTH]
                        P[i,j_stay_south,l]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[H_SOUTH]

                        P[i,j_down_north,l]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[H_NORTH]
                        P[i,J_down_south,l]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[H_SOUTH]

                    case Constants.V_STAY:
                        
                        P[i,j_up,l]=0
                        P[i,j_stay,l]=Constants.P_H_TRANSITION[z_i].P_WIND[H_STAY]
                        P[i,j_down,l]=0

                        P[i,j_up_east,l]=0
                        P[i,j_up_west,l]=0

                        P[i,j_stay_east,l]=Constants.P_H_TRANSITION[z_i].P_WIND[H_EAST]
                        P[i,j_stay_west,l]=Constants.P_H_TRANSITION[z_i].P_WIND[H_WEST]

                        P[i,j_down_east,l]=0
                        P[i,j_down_west,l]=0

                        P[i,j_up_north,l]=0
                        P[i,j_up_south,l]=0

                        P[i, j_stay_north,l]=Constants.P_H_TRANSITION[z_i].P_WIND[H_NORTH]
                        P[i,j_stay_south,l]=Constants.P_H_TRANSITION[z_i].P_WIND[H_SOUTH]

                        P[i,j_down_north,l]=0
                        P[i,J_down_south,l]=0
                    case Constants.V_UP:

                        P[i,j_up,l]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[H_STAY]
                        P[i,j_stay,l]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[H_STAY]
                        P[i,j_down,l]=0

                        P[i,j_up_east,l]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[H_EAST]
                        P[i,j_up_west,l]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[H_WEST]

                        P[i,j_stay_east,l]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[H_EAST]
                        P[i,j_stay_west,l]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[H_WEST]

                        P[i,j_down_east,l]=0
                        P[i,j_down_west,l]=0

                        P[i,j_up_north,l]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[H_NORTH]
                        P[i,j_up_south,l]=Constants.P_V_TRANSITION[1]*Constants.P_H_TRANSITION[z_i].P_WIND[H_SOUTH]

                        P[i, j_stay_north,l]=Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[H_NORTH]
                        P[i,j_stay_south,l]==Constants.P_V_TRANSITION[0]*Constants.P_H_TRANSITION[z_i].P_WIND[H_SOUTH]

                        P[i,j_down_north,l]=0
                        P[i,J_down_south,l]=0



    return P
