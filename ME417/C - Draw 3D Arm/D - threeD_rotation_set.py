import numpy as np

from Z_Every_Function import Rx, Ry, Rz

def threeD_rotation_set(joint_angles, joint_axes):
    """
    Generate a set of 3D rotation matrices corresponding to the angles and axes.
    
    Inputs:
    - joint_angles: 1D array of joint angles
    - joint_axes: list of 'x', 'y', or 'z' that specifies the rotation axis for each joint
    
    Output:
    - R_set: list of 3x3 rotation matrices corresponding to joint angles and axes
    """
    R_set = [None] * len(joint_angles)
    
    for i, axis in enumerate(joint_axes):
        # Check which axis the joint rotates around
        if axis == 'x':
            R_set[i] = Rx(joint_angles[i])
        elif axis == 'y':
            R_set[i] = Ry(joint_angles[i])
        elif axis == 'z':
            R_set[i] = Rz(joint_angles[i])
        else:
            raise ValueError(f"{axis} is not a known joint axis")
    
    return R_set

#test code

link_vectors = [np.array([[1], 
                          [0], 
                          [0]]), 
                np.array([[0.5], 
                          [0], 
                          [0]]), 
                np.array([[0.5], 
                          [0], 
                          [0]])]


joint_angles = [0.4 * np.pi, -0.5 * np.pi, 0.25 * np.pi]
joint_axes = ['z', 'z', 'z']

R_set = threeD_rotation_set(joint_angles, joint_axes)

for idx, R in enumerate(R_set):
    print(f"Rotation matrix {joint_axes[idx]}:")
    print(R)
    print()