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
    """Optimized version of computing the stage cost matrix for the given problem."""
    import itertools
    
    def compute_solar_cost_vectorized(x, t):
        x_sun = np.floor((Constants.M - 1) * (Constants.T - 1 - t) / (Constants.T - 1))
        x_c = x[:, None] + np.array([-1, 0, 1]) * Constants.M
        x_sun = x_sun[:, None]  
        g_solar = np.min((x_c - x_sun) ** 2, axis=1)
        return g_solar

    def compute_cities_cost_vectorized(states, x_city_coordinates, y_city_coordinates):
        z, y, x = states[:, 1], states[:, 2], states[:, 3]
        x_c = x[:, None] + np.array([-1, 0, 1]) * Constants.M
        distance_x = np.sqrt((x_c[:, :, None] - x_city_coordinates) ** 2)
        min_dist_x_sqrd = np.min(distance_x, axis=1)
        cities_cost = np.sqrt(min_dist_x_sqrd ** 2 + (y[:, None] - y_city_coordinates) ** 2) + Constants.LAMBDA_LEVEL * z[:, None]
        g_cities = np.sum(cities_cost, axis=1)
        return g_cities

    t = np.arange(Constants.T)
    z = np.arange(Constants.D)
    y = np.arange(Constants.N)
    x = np.arange(Constants.M)
    state_space = np.array(list(itertools.product(t, z, y, x)))

    K = Constants.T * Constants.D * Constants.N * Constants.M
    input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
    L = len(input_space)
    CITIES_LOCATIONS_array = np.array(Constants.CITIES_LOCATIONS)
    x_city_coordinates = CITIES_LOCATIONS_array[:, 1]
    y_city_coordinates = CITIES_LOCATIONS_array[:, 0]

    
    solar_cost = compute_solar_cost_vectorized(state_space[:, 3], state_space[:, 0])
    cities_cost = compute_cities_cost_vectorized(state_space, x_city_coordinates, y_city_coordinates)


    G = np.ones((K, L)) * np.inf

   
    G[:, Constants.V_STAY] = cities_cost + Constants.LAMBDA_TIMEZONE * solar_cost
    G[:, Constants.V_UP] = cities_cost + Constants.LAMBDA_TIMEZONE * solar_cost
    G[:, Constants.V_DOWN] = cities_cost + Constants.LAMBDA_TIMEZONE * solar_cost


    def map_state_to_index_min_z():
        '''
        Maps a state to the index in the P matrix 
        '''
        
        t_in, y_in, x_in = state_space[:, 0], state_space[:, 2], state_space[:, 3]

        return t_in*(Constants.D*Constants.N*Constants.M) + y_in*Constants.M + x_in
    
    def map_state_to_index_max_z():
        '''
        Maps a state to the index in the P matrix 
        '''
        
        t_in, z_in, y_in, x_in = state_space[:, 0], (Constants.D-1), state_space[:, 2], state_space[:, 3]

        return t_in*(Constants.D*Constants.N*Constants.M) + z_in*(Constants.N*Constants.M) + y_in*Constants.M + x_in

    indices_at_z_zero = map_state_to_index_min_z()
    indices_at_z_max = map_state_to_index_max_z()

    G[indices_at_z_zero, Constants.V_DOWN] = np.inf
    G[indices_at_z_max, Constants.V_UP] = np.inf

    return G

