
import numpy as np


def Ry(phi):
    """
    Rotation matrix about the y-axis.
    
    Input:
    - phi: scalar value for rotation angle
    
    Output:
    - R: 3x3 rotation matrix for rotation by phi around the y-axis
    """
    RM = np.array([[np.cos(phi), 0, np.sin(phi)],
                   [0,           1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])
    
    RM = np.round(RM,8)
    
    return RM


#test code

print(Ry(1))

#[[ 0.54030231  0.          0.84147098]
# [ 0.          1.          0.        ]
# [-0.84147098  0.          0.54030231]]