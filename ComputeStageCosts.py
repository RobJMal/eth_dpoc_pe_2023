# """
#  ComputeStageCosts.py

#  Python function template to compute the stage cost matrix.

#  Dynamic Programming and Optimal Control
#  Fall 2023
#  Programming Exercise
 
#  Contact: Antonio Terpin aterpin@ethz.ch
 
#  Authors: Abhiram Shenoi, Philip Pawlowsky, Antonio Terpin

#  --
#  ETH Zurich
#  Institute for Dynamic Systems and Control
#  --
# """

# import numpy as np


# # def compute_stage_cost(Constants):
# #     """Computes the stage cost matrix for the given problem.

# #     It is of size (K,L) where:
# #         - K is the size of the state space;
# #         - L is the size of the input space; and
# #         - G[i,l] corresponds to the cost incurred when using input l
# #             at state i.

# #     Args:
# #         Constants: The constants describing the problem instance.

# #     Returns:
# #         np.array: Stage cost matrix G of shape (K,L)
# #     """
# #     import itertools

# #     def compute_solar_cost(state_space):
# #         """Computes the solar cost for a given state_space.

# #         Args:
# #             Constants: The constants describing the problem instance.
# #             t: the time space state_space
# #             x: the x coordinate space state_space


# #         Returns:
# #             float: the gsolar cost 
# #         """
# #         x=state_space[3]
# #         t=state_space[0]
# #         x_sun=np.floor((Constants.M-1)*(Constants.T-1-t)/(Constants.T-1))
# #         x_sun=np.full(3,x_sun)
# #         c=np.array([-1,0,1])
# #         x_c=np.full(3,x)+c*Constants.M
# #         g_solar=np.min((x_c-x_sun)**2)
# #         return g_solar
    
# #     def compute_cities_cost(state,x_city_coordinates,y_city_coordinates):
# #         """Computes the connectivity cost for a given state.

# #         Args:
# #             Constants: The constants describing the problem instance.
# #             x: the x coordinate space state
# #             y: the y coordinate space state
# #             z: the z coordinate space state
# #             x_city_coordinates: the set of the x coordinate of the cities
# #             y_city_coordinates: the set of the x coordinate of the cities



# #         Returns:
# #             float: the gcities cost 
# #         """
# #         # t=state[0]
# #         z=state[1]
# #         y=state[2]
# #         x=state[3]
# #         x_c_m1=x-Constants.M
# #         x_c_0=x
# #         x_c_p1=x+Constants.M

# #         distance_x_m1=np.sqrt((x_c_m1-x_city_coordinates)**2)
# #         distance_x_0=np.sqrt((x_c_0-x_city_coordinates)**2)
# #         distance_x_p1=np.sqrt((x_c_p1-x_city_coordinates)**2)
# #         min_dist_x_sqrd = np.minimum(np.minimum(distance_x_m1, distance_x_0), distance_x_p1)

# #         cities_cost=np.sqrt(min_dist_x_sqrd**2 +(y-y_city_coordinates)**2)+Constants.LAMBDA_LEVEL*(z)

# #         g_cities=np.sum(cities_cost)

# #         return g_cities

# #     t = np.arange(0, Constants.T)  
# #     z = np.arange(0, Constants.D)  
# #     y = np.arange(0, Constants.N)  
# #     x = np.arange(0, Constants.M)  
# #     state_space = np.array(list(itertools.product(t, z, y, x)))


# #     K = Constants.T * Constants.D * Constants.N * Constants.M
# #     input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
# #     L = len(input_space)
# #     CITIES_LOCATIONS_array = np.array(Constants.CITIES_LOCATIONS)
# #     x_city_coordinates = CITIES_LOCATIONS_array[:, 1]
# #     y_city_coordinates=CITIES_LOCATIONS_array[:, 0]

# #     G = np.ones((K, L)) * np.inf
# #     for i in range(K):
# #         if(state_space[i][1]==0):
# #             G[i,Constants.V_DOWN]=np.inf
# #             G[i,Constants.V_STAY]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
# #             G[i,Constants.V_UP]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
# #         elif(state_space[i][1]==(Constants.D-1)):
# #             G[i,Constants.V_DOWN]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
# #             G[i,Constants.V_STAY]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
# #             G[i,Constants.V_UP]=np.inf
# #         else:
# #             G[i,Constants.V_DOWN]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
# #             G[i,Constants.V_STAY]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])
# #             G[i,Constants.V_UP]=compute_cities_cost(state_space[i],x_city_coordinates,y_city_coordinates)+Constants.LAMBDA_TIMEZONE*compute_solar_cost(state_space[i])


        
# #     return G

# import numpy as np
# import itertools

# def compute_solar_cost(states, Constants):
#     t = states[:, 0]
#     x = states[:, 3]
#     x_sun = np.floor((Constants.M-1) * (Constants.T-1-t) / (Constants.T-1))
#     x_c = x[:, np.newaxis] + np.array([-1, 0, 1]) * Constants.M
#     g_solar = np.min((x_c - x_sun[:, np.newaxis])**2, axis=1)
#     return g_solar

# def compute_cities_cost(states, x_city_coordinates, y_city_coordinates, Constants):
#     z = states[:, 1]
#     y = states[:, 2]
#     x = states[:, 3]
#     x_c = x[:, np.newaxis] + np.array([-Constants.M, 0, Constants.M])

#     # Calculate squared distances for each x_c to city x-coordinates
#     distance_x = np.sqrt((x_c[:, :, np.newaxis] - x_city_coordinates)**2)
#     min_dist_x_sqrd = np.min(distance_x, axis=1)

#     # Calculate squared distances for y to city y-coordinates
#     distance_y = np.sqrt((y[:, np.newaxis] - y_city_coordinates)**2)

#     # Combine x and y distances to compute cities cost
#     cities_cost = np.sqrt(min_dist_x_sqrd + distance_y**2)
#     g_cities = np.sum(cities_cost, axis=1) + Constants.LAMBDA_LEVEL * z
#     return g_cities

# def compute_stage_cost(Constants):
#     # Generate state space
#     t = np.arange(Constants.T)  
#     z = np.arange(Constants.D)  
#     y = np.arange(Constants.N)  
#     x = np.arange(Constants.M)  
#     state_space = np.array(list(itertools.product(t, z, y, x)))

#     K = Constants.T * Constants.D * Constants.N * Constants.M
#     L = 3  # Number of actions: Constants.V_DOWN, Constants.V_STAY, Constants.V_UP
#     CITIES_LOCATIONS_array = np.array(Constants.CITIES_LOCATIONS)
#     x_city_coordinates = CITIES_LOCATIONS_array[:, 1]
#     y_city_coordinates = CITIES_LOCATIONS_array[:, 0]

#     # Compute costs
#     g_solar = compute_solar_cost(state_space, Constants)
#     g_cities = compute_cities_cost(state_space, x_city_coordinates, y_city_coordinates, Constants)

#     # Initialize stage cost matrix
#     G = np.ones((K, L)) * np.inf

#     # Populate G with computed costs
#     valid_indices = np.arange(K)
#     G[valid_indices, Constants.V_STAY] = g_cities + Constants.LAMBDA_TIMEZONE * g_solar
#     G[valid_indices, Constants.V_UP] = g_cities + Constants.LAMBDA_TIMEZONE * g_solar
#     G[valid_indices, Constants.V_DOWN] = g_cities + Constants.LAMBDA_TIMEZONE * g_solar

#     # Boundary conditions
#     z_indices = state_space[:, 1]
#     G[z_indices == 0, Constants.V_DOWN] = np.inf
#     G[z_indices == (Constants.D - 1), Constants.V_UP] = np.inf

#     return G

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

    G[indices_at_z_zero, Constants.V_DOWN] = np.inf
    G[indices_at_z_max, Constants.V_UP] = np.inf

    return G

