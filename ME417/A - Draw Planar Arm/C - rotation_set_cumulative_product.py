# Cumulative product of rotation matrices

import numpy as np
import Z_Every_Function as F

def rotation_set_cumulative_product(R_set):
    """
    Take the cumulative product of a set of rotation matrices.
    
    Input:
    - R_set: A 1xn or nx1 cell array, each element of which is a 2x2 or 3x3 rotation matrix
    
    Output:
    - R_set_c: A 1xn or nx1 cell array, the ith element of which is a 2x2 or 3x3 rotation
                matrix that is the cumulative product of the rotation matrices in
                R_set from the first matrix up to the ith matrix
    """
    R_set_c = R_set.copy()
    
    # Loop over R_set_c, multiplying each matrix into the one after it
    for i in range(1, len(R_set_c)):
        R_set_c[i] = np.dot(R_set_c[i - 1], R_set_c[i])
    
    return R_set_c



#test code

R_set = np.array([0, np.pi, np.pi / 4])
print(R_set)
s = F.planar_rotation_set(R_set)
s_c = rotation_set_cumulative_product(s)

for i, R in enumerate(s_c):
    print(f"Cumulative rotation matrix {i + 1}:\n{R}\n")


R_set = np.array([[0],
         [np.pi],
         [np.pi / 4]])

print(R_set)

s2 = F.planar_rotation_set(R_set)
s_c2 = rotation_set_cumulative_product(s2)

for i, R in enumerate(s_c2):
    print(f"Cumulative rotation matrix {i + 1} for s2:\n{R}\n")


    '''R_set = [0.         3.14159265 0.78539816]

Cumulative rotation matrix 1:
[[ 1. -0.]
 [ 0.  1.]]

Cumulative rotation matrix 2:
[[-1.  0.]
 [ 0. -1.]]

Cumulative rotation matrix 3:
[[-0.70710678  0.70710678]
 [-0.70710678 -0.70710678]]

R_set = [[0.        ]
 [3.14159265]
 [0.78539816]]

Cumulative rotation matrix 1 for s2:
[[ 1. -0.]
 [ 0.  1.]]

Cumulative rotation matrix 2 for s2:
[[-1.  0.]
 [ 0. -1.]]

Cumulative rotation matrix 3 for s2:
[[-0.70710678  0.70710678]
 [-0.70710678 -0.70710678]]'''