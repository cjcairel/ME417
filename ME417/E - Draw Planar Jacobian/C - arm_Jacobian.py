import numpy as np
import sympy as sp
import Z_Every_Function as F


def arm_Jacobian(link_vectors, joint_angles, joint_axes, link_number):
    """
    Construct a Jacobian for a chain of links as a function of the link vectors, 
    the joint angles, joint axes, and the number of the link whose endpoint is 
    the location where the Jacobian is evaluated.
    
    Parameters:
    -----------
    link_vectors : list of numpy arrays
        Each element is a 3x1 link vector.
        
    joint_angles : list or numpy array
        Each element is the joint angle preceding the corresponding link.
        
    joint_axes : list of strings
        Each element is 'x', 'y', or 'z', designating the axis of the corresponding joint.
        
    link_number : int
        The number of the link whose Jacobian we want to evaluate.

    Returns:
    --------
    J : sympy.Matrix or numpy.ndarray
        The Jacobian for the end of link 'link_number'.
    
    Additional outputs (for debugging and verification):
    ----------------------------------------------------
    link_end_set_with_base : list of numpy arrays
    v_diff : list of numpy arrays
    joint_axis_vectors : list of numpy arrays
    joint_axis_vectors_R : list of numpy arrays
    R_links : list of numpy arrays
    """
    
    # Step 1: Get link endpoints, rotation matrices for each joint, and link cumulative orientations
    (link_ends, R_joints, R_links, link_vectors_in_world,
      link_end_set, link_end_set_with_base) = F.threeD_robot_arm_endpoints(link_vectors, joint_angles, joint_axes)
    
    # Step 2: Vector difference from each link endpoint to the end of the specified link
    v_diff = F.vector_set_difference(link_end_set_with_base[link_number], link_end_set_with_base)
    
    # Step 3: Generate joint axis vectors in local coordinates
    joint_axis_vectors = F.threeD_joint_axis_set(joint_axes)
    
    # Step 4: Rotate joint axes into world coordinates
    joint_axis_vectors_R = F.vector_set_rotate(joint_axis_vectors, R_links)
    
    # Step 5: Initialize Jacobian as a zero matrix
    J = np.zeros((3, len(joint_angles)), dtype=object)
    
    # Step 6: Fill in the Jacobian matrix up to the specified link
    for i in range(link_number):
        J[:, i] = np.cross(joint_axis_vectors_R[i].flatten(), v_diff[i].flatten())
    
    return J, link_ends, link_end_set, link_end_set_with_base, v_diff, joint_axis_vectors, joint_axis_vectors_R, R_links

# Test code
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
link_number = 3
(J_numeric, link_ends, link_end_set, link_end_set_with_base, 
 v_diff, joint_axis_vectors, joint_axis_vectors_R, R_links) = arm_Jacobian(link_vectors, joint_angles, joint_axes, link_number)


print(J_numeric)

'''

[[-1.0235432773455067 -0.07248675734550669 -0.22699525234550666]
 [1.2300485111096988 0.9210315211096989 0.44550326110969884]
 [0.0 0.0 0.0]]
 
 '''