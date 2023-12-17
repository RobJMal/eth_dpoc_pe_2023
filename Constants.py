"""
 Constants.py

 Python script containg the definition of the class Constants
 that holds all the problem constants.

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


class Wind:
    """
    Probability of disturbances due to wind at the same altitude.
    self.P_WIND is indexed by the directions defined in Constants
    """

    DIR = 0
    P_WIND = [0, 0, 0, 0, 0]

    def __init__(self, w_dir, p_dom):
        self.DIR = w_dir
        p_other = (1 - p_dom) / 4

        # Error checking
        if not np.isclose(p_dom + 4 * p_other, 1):
            print("[ERROR] Not a valid PDF over the wind directions")
        if w_dir not in range(5):
            print(
                "[ERROR] Wind direction not valid, should be NORTH (0), EAST (1), \
                    SOUTH (2), WEST (3)"
            )

        self.P_WIND = [p_other] * 5
        self.P_WIND[self.DIR] = p_dom


class Constants:
    # ----- Movement definitions -----
    # [!] Do not change.
    
    # Movement within a plane and wind directions
    H_NORTH = 0
    H_EAST = 1
    H_SOUTH = 2
    H_WEST = 3
    H_STAY = 4
    
    # Vertical movements
    V_DOWN = 0
    V_STAY = 1
    V_UP   = 2

    # ----- Constants -----
    # Feel free to tweak these to test your solution.

    # State space constants
    N = 15  # Size of the y axis (north to south)
    M = 20  # Size of the x axis (west to east)
    D = 4  # Size of the z axis (bottom to top)
    T = 10   # Number of time subdivisions

    # Map constants
    N_CITIES = 1  # Number of cities


    # ----- Factors -----

    # Stage cost factors
    LAMBDA_LEVEL    = 0.4  # Factor in the cost related to the balloon altitude
    LAMBDA_TIMEZONE = 0.4  # Factor in the cost related to the sun position

    # Discount factor
    ALPHA = 0.99
    

    # ----- Disturbances -----
    # Stochastic disturbances at the same altitude.
    # To access the probability of moving NORTH at level z = 2,
    # P_H_TRANSITION[2].P_WIND[H_NORTH]
    P_H_TRANSITION = [
        Wind(H_WEST, 0.9),  # Level 0
        Wind(H_NORTH, 0.6), # Level 1
        Wind(H_EAST, 0.9),  # Level 2
        Wind(H_WEST, 0.2),  # Level 3
    ]

    # Stochastic disturbances related to change in altitude.
    P_V_TRANSITION = [0.1, 0.9]  # [V_STAY, V_UP or V_DOWN]

    # Locations of the cities
    # [(y,x),(y,x),...] of the cities
    # Filled by GenerateMap.py
    CITIES_LOCATIONS = np.stack((np.random.randint(0,N,N_CITIES),
                                    np.random.randint(0,M,N_CITIES)), axis=1).tolist()
