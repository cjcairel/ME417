
import numpy as np


def Rz(theta):
    """
    Rotation matrix about the z-axis.
    
    Input:
    - theta: scalar value for rotation angle
    
    Output:
    - R: 3x3 rotation matrix for rotation by theta around the z-axis
    """
    RM = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    
    RM = np.round(RM, 8)
    
    return RM

#test code 
print(Rz(1))

#[[ 0.54030231 -0.84147098  0.        ]
# [ 0.84147098  0.54030231  0.        ]
# [ 0.          0.          1.        ]]