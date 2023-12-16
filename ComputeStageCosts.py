"""
 ComputeStageCosts.py

 Python function template to compute the stage cost matrix.

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


def compute_stage_cost(Constants):
    """Computes the stage cost matrix for the given problem.

    It is of size (K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - G[i,l] corresponds to the cost incurred when using input l
            at state i.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Stage cost matrix G of shape (K,L)
    """
    import itertools

    def compute_solar_cost(state_space):
        """Computes the solar cost for a given state_space.

        Args:
            Constants: The constants describing the problem instance.
            t: the time space state_space
            x: the x coordinate space state_space


        Returns:
            float: the gsolar cost 
        """
        x=state_space[3]
        t=state_space[0]
        x_sun=np.floor((Constants.M-1)*(Constants.T-1-t)/(Constants.T-1))
        x_sun=np.full(3,x_sun)
        c=np.array([-1,0,1])
        x_c=np.full(3,x)+c*Constants.M
        g_solar=np.min((x_c-x_sun)**2)
        return g_solar
    
    def compute_cities_cost(state,x_city_coordinates,y_city_coordinates):
        """Computes the connectivity cost for a given state.

        Args:
            Constants: The constants describing the problem instance.
            x: the x coordinate space state
            y: the y coordinate space state
            z: the z coordinate space state
            x_city_coordinates: the set of the x coordinate of the cities
            y_city_coordinates: the set of the x coordinate of the cities



        Returns:
            float: the gcities cost 
        """
        # t=state[0]
        z=state[1]
        y=state[2]
        x=state[3]
        x_c_m1=x-Constants.M
        x_c_0=x
        x_c_p1=x+Constants.M

        distance_x_m1=np.sqrt((x_c_m1-x_city_coordinates)**2)
        distance_x_0=np.sqrt((x_c_0-x_city_coordinates)**2)
        distance_x_p1=np.sqrt((x_c_p1-x_city_coordinates)**2)
        min_dist_x_sqrd = np.minimum(np.minimum(distance_x_m1, distance_x_0), distance_x_p1)

        cities_cost=np.sqrt(min_dist_x_sqrd**2 +(y-y_city_coordinates)**2)+Constants.LAMBDA_LEVEL*(z)

        g_cities=np.sum(cities_cost)

        return g_cities

    t = np.arange(0, Constants.T)  
    z = np.arange(0, Constants.D)  
    y = np.arange(0, Constants.N)  
    x = np.arange(0, Constants.M)  
    state_space = np.array(list(itertools.product(t, z, y, x)))


    K = Constants.T * Constants.D * Constants.N * Constants.M
    input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
    L = len(input_space)
    CITIES_LOCATIONS_array = np.array(Constants.CITIES_LOCATIONS)
    x_city_coordinates = CITIES_LOCATIONS_array[:, 1]
    y_city_coordinates=CITIES_LOCATIONS_array[:, 0]

    G = np.ones((K, L)) * np.inf
    for i in range(K):
        if(state_space[i][1]==0):
            G[i,Constants.V_DOWN]=np.inf
            G[i,Constants.V_STAY]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
            G[i,Constants.V_UP]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
        elif(state_space[i][1]==(Constants.D-1)):
            G[i,Constants.V_DOWN]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
            G[i,Constants.V_STAY]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
            G[i,Constants.V_UP]=np.inf
        else:
            G[i,Constants.V_DOWN]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
            G[i,Constants.V_STAY]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
            G[i,Constants.V_UP]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])


        
    return G

