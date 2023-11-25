import numpy as np
import itertools

t = np.arange(0, 5)  
z = np.arange(0, 4)  
y = np.arange(0, 5)  
x = np.arange(0, 5)  
state_space = np.array(list(itertools.product(t, z, y, x)))
print (state_space.shape)
print(state_space[499])
print(state_space[99])
state_index = np.where((state_space == (2, 2, 2, 2)).all(axis=1))[0][0]
print(state_index)