import numpy as np



def threeD_joint_axis_set(joint_axes):
    """
    Generate a set of unit vectors along specified x, y, or z axes.
    
    Parameters:
    -----------
    joint_axes : list of str
        Each element is a string 'x', 'y', or 'z' specifying an axis of rotation.
        
    Returns:
    --------
    joint_axis_vectors : list of numpy arrays
        A list of unit vectors corresponding to the specified axes in joint_axes.
    """
    
    # Initialize joint_axis_vectors to be the same size as joint_axes
    joint_axis_vectors = [None] * len(joint_axes)
    
    # Loop over the joint axes
    for idx in range(len(joint_axes)):
        axis = joint_axes[idx]
        if axis == 'x':
            joint_axis_vectors[idx] = np.array([[1], [0], [0]])
        elif axis == 'y':
            joint_axis_vectors[idx] = np.array([[0], [1], [0]])
        elif axis == 'z':
            joint_axis_vectors[idx] = np.array([[0], [0], [1]])
        else:
            raise ValueError(f'{axis} is not a known joint axis')
    
    return joint_axis_vectors


#test code
joint_axes = ['x', 'y', 'z']

s = threeD_joint_axis_set(joint_axes)

for i in s:
    print(i)
    print()