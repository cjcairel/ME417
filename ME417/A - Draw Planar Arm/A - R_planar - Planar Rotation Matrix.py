#Planar Rotation Matrix

import numpy as np

def R_planar(theta):
    """
    Planar rotation matrix.

    Parameters:
        theta (float): Scalar value for rotation angle in radians.

    Returns:
        array: 2x2 rotation matrix for rotation by theta.
    """
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    R = np.round(R,8)
    return R


#test code
theta = np.pi/4
R = R_planar(theta)

print(R)

#   should be [[ 0.70710678 -0.70710678]
#              [ 0.70710678  0.70710678]]

