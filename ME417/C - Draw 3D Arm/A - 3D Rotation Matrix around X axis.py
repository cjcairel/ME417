import numpy as np

def Rx(psi):
    """
    Rotation matrix about the x-axis.
    
    Input:
    - psi: scalar value for rotation angle

    Output:
    - R: 3x3 rotation matrix for rotation by psi around the x-axis
    """
    RM = np.array([[1, 0, 0],
                [0, np.cos(psi), -np.sin(psi)],
                [0, np.sin(psi), np.cos(psi)]])
    
    RM = np.round(RM,8)

    return RM


#test code 
print(Rx(1))

#[[ 1.          0.          0.        ]
# [ 0.          0.54030231 -0.84147098]
# [ 0.          0.84147098  0.54030231]]