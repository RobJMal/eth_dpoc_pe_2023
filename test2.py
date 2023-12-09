"""
 test.py

 Python script implementing test cases for debugging.

 Dynamic Programming and Optimal Control
 Fall 2023
 Programming Exercise

 Contact: Antonio Terpin aterpin@ethz.ch
 
 Authors: Abhiram Shenoi, Philip Pawlowsky

 --
 ETH Zurich
 Institute for Dynamic Systems and Control
 --
"""

import numpy as np
from ComputeStageCosts import compute_stage_cost
from ComputeTransitionProbabilities import compute_transition_probabilities
from Constants import Constants
from Solver import solution, freestyle_solution
import pickle
import itertools
import tracemalloc

# Additional imports
import cProfile

if __name__ == "__main__":
    n_tests = 1 # 3
    for i in range(n_tests):
        i=2
        print("-----------")
        print("Test " + str(i))
        with open("tests/test" + str(i) + ".pkl", "rb") as f:
            loaded_constants = pickle.load(f)
            for attr_name, attr_value in loaded_constants.items():
                x=0
                # if hasattr(Constants, attr_name):
                #     setattr(Constants, attr_name, attr_value)

        file = np.load("tests/test" + str(i) + ".npz")

        # State space
        t = np.arange(0, Constants.T)
        z = np.arange(0, Constants.D)
        y = np.arange(0, Constants.N)
        x = np.arange(0, Constants.M)
        state_space = np.array(list(itertools.product(t, z, y, x)))

        # Begin tests
        K = len(state_space)
        tracemalloc.start()
        P = compute_transition_probabilities(Constants)
        current,peak=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(peak/2**20)

        if not np.all(
            np.logical_or(np.isclose(P.sum(axis=1), 1), np.isclose(P.sum(axis=1), 0))
        ):
            print(
                "[ERROR] Transition probabilities do not sum up to 1 or 0 along axis 1!"
            )

        G = compute_stage_cost(Constants)

        passed = True
        if False:#not np.allclose(P, file["P"], rtol=1e-4, atol=1e-7):
            # P2=file["P"]-P
            # non_zero_indices = np.nonzero(P2)
            # for index in zip(*non_zero_indices):
            #     print(f"P{index} = {P2[index]}")
            print("Wrong transition probabilities")
            # print(P[443,38,1])
            passed = False
        else:
            print("Correct transition probabilities")

        if False:#not np.allclose(G, file["G"], rtol=1e-4, atol=1e-7):
            print(G[498][1],file["G"][498][1])
            print("Wrong stage costs")
            # G2=file["G"]-G
            # non_zero_indices = np.nonzero(G2)
            # for index in zip(*non_zero_indices):
            #     print(f"G{index} = {G[index]}")
            passed = False
        else:
            print("Correct stage costs")

        # normal solution
        
        [J_opt, u_opt] = solution(P, G, Constants.ALPHA)

     
        if False:#not np.allclose(J_opt, file["J"], rtol=1e-4, atol=1e-7):
            print("[guided solution] Wrong optimal cost")
            passed = False
        else:
            print("[guided solution] Correct optimal cost")

        # # freestyle solution
        # tracemalloc.start()
        # [J_opt, u_opt] = freestyle_solution(Constants)
        # current,peak=tracemalloc.get_traced_memory()
        # tracemalloc.stop()
        # print(peak/2**20)
        # if not np.allclose(J_opt, file["J"], rtol=1e-4, atol=1e-7):
        #     print("[freestyle solution] Wrong optimal cost")
        #     passed = False
        # else:
        #     print("[freestyle solution] Correct optimal cost")

        # Checking time for optimization
        cprofile_function_name = f'compute_transition_probabilities(Constants)'
        cprofile_file_name = 'optimization/max_test_output_file_' + str(i) + '.prof'
        cProfile.run(cprofile_function_name, cprofile_file_name)

    print("-----------")