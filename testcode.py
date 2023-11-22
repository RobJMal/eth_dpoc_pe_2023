import numpy as np
import itertools

t = np.arange(0, 5)  
z = np.arange(0, 6)  
y = np.arange(0, 3)  
x = np.arange(0, 4)  
state_space = np.array(list(itertools.product(t, z, y, x)))
print (state_space.shape)
print(state_space[359])
state_index = np.where((state_space == (4, 5, 1, 3)).all(axis=1))[0][0]
print(state_index)