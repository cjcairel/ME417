import numpy as np
import Z_Every_Function as F

def planar_rotation_set(joint_angles):
    """
    Generate a set of planar rotation matrices corresponding to the angles in the input vector.

    Parameters:
        joint_angles: a 1xn or nx1 vector of joint angles

    Returns:
        R_set: a list of planar rotation matrices for each angle in the vector
    """
    
    joint_angles = np.ravel(joint_angles)

    # Create an empty list for rotation matrices
    R_set = []
    
    # Loop over joint angles to create and store each rotation matrix
    for theta in joint_angles:
        R = F.R_planar(theta)
        R_set.append(R.reshape(2, 2))
        
    return R_set

# Test code
R_set = np.array([0, np.pi, np.pi/4])
s = planar_rotation_set(R_set)

for i, R in enumerate(s):
    print(f"Rotation matrix for angle {i + 1}:\n{R}\n")

R_set = np.array([  [0],
                    [np.pi],
                    [np.pi/4]   ])

s = planar_rotation_set(R_set)

for i, R in enumerate(s):
    print(f"Rotation matrix for angle s2 - {i + 1}:\n{R}\n")

'''Rotation matrix for angle 1:
[[ 1. -0.]
 [ 0.  1.]]

Rotation matrix for angle 2:
[[-1. -0.]
 [ 0. -1.]]

Rotation matrix for angle 3:
[[ 0.70710678 -0.70710678]
 [ 0.70710678  0.70710678]]

Rotation matrix for angle s2 - 1:
[[ 1. -0.]
 [ 0.  1.]]

Rotation matrix for angle s2 - 2:
[[-1. -0.]
 [ 0. -1.]]

Rotation matrix for angle s2 - 3:
[[ 0.70710678 -0.70710678]
 [ 0.70710678  0.70710678]]'''