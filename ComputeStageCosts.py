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


# def compute_stage_cost(Constants):
#     """Computes the stage cost matrix for the given problem.

#     It is of size (K,L) where:
#         - K is the size of the state space;
#         - L is the size of the input space; and
#         - G[i,l] corresponds to the cost incurred when using input l
#             at state i.

#     Args:
#         Constants: The constants describing the problem instance.

#     Returns:
#         np.array: Stage cost matrix G of shape (K,L)
#     """
#     import itertools

#     def compute_solar_cost(state_space):
#         """Computes the solar cost for a given state_space.

#         Args:
#             Constants: The constants describing the problem instance.
#             t: the time space state_space
#             x: the x coordinate space state_space


#         Returns:
#             float: the gsolar cost 
#         """
#         x=state_space[3]
#         t=state_space[0]
#         x_sun=np.floor((Constants.M-1)*(Constants.T-1-t)/(Constants.T-1))
#         x_sun=np.full(3,x_sun)
#         c=np.array([-1,0,1])
#         x_c=np.full(3,x)+c*Constants.M
#         g_solar=np.min((x_c-x_sun)**2)
#         return g_solar
    
#     def compute_cities_cost(state,x_city_coordinates,y_city_coordinates):
#         """Computes the connectivity cost for a given state.

#         Args:
#             Constants: The constants describing the problem instance.
#             x: the x coordinate space state
#             y: the y coordinate space state
#             z: the z coordinate space state
#             x_city_coordinates: the set of the x coordinate of the cities
#             y_city_coordinates: the set of the x coordinate of the cities



#         Returns:
#             float: the gcities cost 
#         """
#         # t=state[0]
#         z=state[1]
#         y=state[2]
#         x=state[3]
#         x_c_m1=x-Constants.M
#         x_c_0=x
#         x_c_p1=x+Constants.M

#         distance_x_m1=np.sqrt((x_c_m1-x_city_coordinates)**2)
#         distance_x_0=np.sqrt((x_c_0-x_city_coordinates)**2)
#         distance_x_p1=np.sqrt((x_c_p1-x_city_coordinates)**2)
#         min_dist_x_sqrd = np.minimum(np.minimum(distance_x_m1, distance_x_0), distance_x_p1)

#         cities_cost=np.sqrt(min_dist_x_sqrd**2 +(y-y_city_coordinates)**2)+Constants.LAMBDA_LEVEL*(z)

#         g_cities=np.sum(cities_cost)

#         return g_cities

#     t = np.arange(0, Constants.T)  
#     z = np.arange(0, Constants.D)  
#     y = np.arange(0, Constants.N)  
#     x = np.arange(0, Constants.M)  
#     state_space = np.array(list(itertools.product(t, z, y, x)))


#     K = Constants.T * Constants.D * Constants.N * Constants.M
#     input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
#     L = len(input_space)
#     CITIES_LOCATIONS_array = np.array(Constants.CITIES_LOCATIONS)
#     x_city_coordinates = CITIES_LOCATIONS_array[:, 1]
#     y_city_coordinates=CITIES_LOCATIONS_array[:, 0]

#     G = np.ones((K, L)) * np.inf
#     for i in range(K):
#         if(state_space[i][1]==0):
#             G[i,Constants.V_DOWN]=np.inf
#             G[i,Constants.V_STAY]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
#             G[i,Constants.V_UP]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
#         elif(state_space[i][1]==(Constants.D-1)):
#             G[i,Constants.V_DOWN]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
#             G[i,Constants.V_STAY]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
#             G[i,Constants.V_UP]=np.inf
#         else:
#             G[i,Constants.V_DOWN]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
#             G[i,Constants.V_STAY]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
#             G[i,Constants.V_UP]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])

#     return G


def compute_stage_cost(Constants):
    """Optimized version of computing the stage cost matrix for the given problem."""
    
    def compute_solar_cost_vectorized(x, t):
        x_sun = np.floor((Constants.M - 1) * (Constants.T - 1 - t) / (Constants.T - 1))
        x_c = x[:, None] + np.array([-1, 0, 1]) * Constants.M
        x_sun = x_sun[:, None]  # Add an extra dimension to x_sun for broadcasting
        g_solar = np.min((x_c - x_sun) ** 2, axis=1)
        # print(g_solar.shape)
        return g_solar

    def compute_cities_cost_vectorized(states, x_city_coordinates, y_city_coordinates):
        z, y, x = states[:, 1], states[:, 2], states[:, 3]
        x_c = x[:, None] + np.array([-1, 0, 1]) * Constants.M
        distance_x = np.sqrt((x_c[:, :, None] - x_city_coordinates) ** 2)
        min_dist_x_sqrd = np.min(distance_x, axis=1)
        cities_cost = np.sqrt(min_dist_x_sqrd ** 2 + (y[:, None] - y_city_coordinates) ** 2) + Constants.LAMBDA_LEVEL * z[:, None]
        g_cities = np.sum(cities_cost, axis=1)
        return g_cities

    # Create the state space grid
    t = np.arange(Constants.T)
    z = np.arange(Constants.D)
    y = np.arange(Constants.N)
    x = np.arange(Constants.M)
    state_space = np.array(np.meshgrid(t, z, y, x)).T.reshape(-1, 4)

    K = Constants.T * Constants.D * Constants.N * Constants.M
    input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
    L = len(input_space)
    CITIES_LOCATIONS_array = np.array(Constants.CITIES_LOCATIONS)
    x_city_coordinates = CITIES_LOCATIONS_array[:, 1]
    y_city_coordinates = CITIES_LOCATIONS_array[:, 0]

    # Compute costs
    solar_cost = compute_solar_cost_vectorized(state_space[:, 3], state_space[:, 0])
    cities_cost = compute_cities_cost_vectorized(state_space, x_city_coordinates, y_city_coordinates)

    # Initialize cost matrix
    G = np.ones((K, L)) * np.inf

    # Vectorized computation for the cost matrix
    G[:, Constants.V_STAY] = cities_cost + Constants.LAMBDA_TIMEZONE * solar_cost
    G[:, Constants.V_UP] = cities_cost + Constants.LAMBDA_TIMEZONE * solar_cost
    G[:, Constants.V_DOWN] = cities_cost + Constants.LAMBDA_TIMEZONE * solar_cost

    # Handle boundary conditions
    def map_state_to_index_min_z():
        '''
        Maps a state to the index in the P matrix 
        '''
        import Constants
        t_in, y_in, x_in = state_space[:, 0], state_space[:, 2], state_space[:, 3]

        return t_in*(Constants.Constants.D*Constants.Constants.N*Constants.Constants.M) + y_in*Constants.Constants.M + x_in
    
    def map_state_to_index_max_z():
        '''
        Maps a state to the index in the P matrix 
        '''
        import Constants
        t_in, z_in, y_in, x_in = state_space[:, 0], (Constants.Constants.D-1), state_space[:, 2], state_space[:, 3]

        return t_in*(Constants.Constants.D*Constants.Constants.N*Constants.Constants.M) + z_in*(Constants.Constants.N*Constants.Constants.M) + y_in*Constants.Constants.M + x_in

    indices_at_z_zero = map_state_to_index_min_z()
    indices_at_z_max = map_state_to_index_max_z()

    # z = state_space[:, 1]
    # G[value, Constants.V_DOWN] = np.inf    # whereever the statespace has z equal to 0 
    # G[z == Constants.D - 1, Constants.V_UP] = np.inf    # whereever the statespace has z equal to Constants.D-1


    G[indices_at_z_zero, Constants.V_DOWN] = np.inf
    G[indices_at_z_max, Constants.V_UP] = np.inf

    return G

# # Example usage
# import itertools

# class ConstantsExample:
#     T, D, N, M = 3, 3, 3, 3
#     V_DOWN, V_STAY, V_UP = 0, 1, 2
#     LAMBDA_TIMEZONE = 0.1
#     LAMBDA_LEVEL = 0.2
#     CITIES_LOCATIONS = [(1, 2), (2, 2)]

# t = np.arange(0, ConstantsExample.T)  
# z = np.arange(0, ConstantsExample.D)  
# y = np.arange(0, ConstantsExample.N)  
# x = np.arange(0, ConstantsExample.M)  
# state_space = np.array(list(itertools.product(t, z, y, x)))


# K = ConstantsExample.T * ConstantsExample.D * ConstantsExample.N * ConstantsExample.M
# input_space = np.array([ConstantsExample.V_DOWN, ConstantsExample.V_STAY, ConstantsExample.V_UP])
# L = len(input_space)
# CITIES_LOCATIONS_array = np.array(ConstantsExample.CITIES_LOCATIONS)
# x_city_coordinates = CITIES_LOCATIONS_array[:, 1]
# y_city_coordinates=CITIES_LOCATIONS_array[:, 0]

# def compute_solar_cost(state_space):
#         """Computes the solar cost for a given state_space.

#         Args:
#             Constants: The constants describing the problem instance.
#             t: the time space state_space
#             x: the x coordinate space state_space


#         Returns:
#             float: the gsolar cost 
#         """
#         x=state_space[3]
#         t=state_space[0]
#         x_sun=np.floor((ConstantsExample.M-1)*(ConstantsExample.T-1-t)/(ConstantsExample.T-1))
#         x_sun=np.full(3,x_sun)
#         c=np.array([-1,0,1])
#         x_c=np.full(3,x)+c*ConstantsExample.M
#         g_solar=np.min((x_c-x_sun)**2)
#         return g_solar

# def compute_solar_cost_vectorized(x, t):
#         x_sun = np.floor((ConstantsExample.M - 1) * (ConstantsExample.T - 1 - t) / (ConstantsExample.T - 1))
#         x_c = x[:, None] + np.array([-1, 0, 1]) * ConstantsExample.M
#         x_sun = x_sun[:, None]  # Add an extra dimension to x_sun for broadcasting
#         g_solar = np.min((x_c - x_sun) ** 2, axis=1)
#         # print(g_solar.shape)
#         return g_solar

# def compute_cities_cost_vectorized(states, x_city_coordinates, y_city_coordinates):
#         z, y, x = states[:, 1], states[:, 2], states[:, 3]
#         x_c = x[:, None] + np.array([-1, 0, 1]) * ConstantsExample.M
#         distance_x = np.sqrt((x_c[:, :, None] - x_city_coordinates) ** 2)
#         min_dist_x_sqrd = np.min(distance_x, axis=1)
#         cities_cost = np.sqrt(min_dist_x_sqrd ** 2 + (y[:, None] - y_city_coordinates) ** 2) + ConstantsExample.LAMBDA_LEVEL * z[:, None]
#         g_cities = np.sum(cities_cost, axis=1)
#         return g_cities

# def compute_cities_cost(state,x_city_coordinates,y_city_coordinates):
#         """Computes the connectivity cost for a given state.

#         Args:
#             Constants: The constants describing the problem instance.
#             x: the x coordinate space state
#             y: the y coordinate space state
#             z: the z coordinate space state
#             x_city_coordinates: the set of the x coordinate of the cities
#             y_city_coordinates: the set of the x coordinate of the cities



#         Returns:
#             float: the gcities cost 
#         """
#         # t=state[0]
#         z=state[1]
#         y=state[2]
#         x=state[3]
#         x_c_m1=x-ConstantsExample.M
#         x_c_0=x
#         x_c_p1=x+ConstantsExample.M

#         distance_x_m1=np.sqrt((x_c_m1-x_city_coordinates)**2)
#         distance_x_0=np.sqrt((x_c_0-x_city_coordinates)**2)
#         distance_x_p1=np.sqrt((x_c_p1-x_city_coordinates)**2)
#         min_dist_x_sqrd = np.minimum(np.minimum(distance_x_m1, distance_x_0), distance_x_p1)

#         cities_cost=np.sqrt(min_dist_x_sqrd**2 +(y-y_city_coordinates)**2)+ConstantsExample.LAMBDA_LEVEL*(z)

#         g_cities=np.sum(cities_cost)

#         return g_cities

# # solar_cost_vectorized = compute_solar_cost_vectorized(state_space[:, 3], state_space[:, 0])
# # solar_cost_list = []

# # for i in range(K):
# #     solar_cost = compute_solar_cost(state_space[i])
# #     solar_cost_list.append(solar_cost)

# # solar_cost_list = np.array(solar_cost_list)

# # print(solar_cost_vectorized - solar_cost_list)


# # city_cost_vectorized = compute_cities_cost_vectorized(state_space, x_city_coordinates, y_city_coordinates)
# # city_cost_list = []

# # for i in range(K):
# #     city_cost = compute_cities_cost(state_space[i], x_city_coordinates, y_city_coordinates)
# #     city_cost_list.append(city_cost)

# # city_cost_list = np.array(city_cost_list)

# # print(np.nonzero(city_cost_vectorized - city_cost_list))

# # Test the function with example constants
# constants = ConstantsExample()
# # G = compute_stage_cost(constants)
# G_optimized = compute_stage_cost_optimized(constants)

# # diff_G = G - G_optimized
# # non_zero_indices = np.nonzero(diff_G)
# # for index in zip(*non_zero_indices):
# #     print(f"G{index} = {G[index]}")

# # print()
# # # print(G)
# # print()
# # print(G_optimized)

# def map_state_to_index(input_state):
#     '''
#     Maps a state to the index in the P matrix 
#     '''
#     t_in, z_in, y_in, x_in = input_state[0], 0, input_state[2], input_state[3]

#     return t_in*(Constants.Constants.D*Constants.Constants.N*Constants.Constants.M) + z_in*(Constants.Constants.N*Constants.Constants.M) + y_in*Constants.Constants.M + x_in
