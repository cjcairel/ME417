import numpy as np

import Z_Every_Function as f


def threeD_robot_arm_endpoints(link_vectors, joint_angles, joint_axes):
    """
    Compute the endpoints of a 3D robot arm based on link vectors and joint angles.
    
    Inputs:
    - link_vectors: list of 3x1 vectors describing the links
    - joint_angles: list of joint angles preceding each link
    - joint_axes: list of 'x', 'y', or 'z' defining the rotation axis for each joint
    
    Outputs:
    - link_ends: 3x(n+1) matrix of link endpoints, starting from the origin
    """
    # Generate rotation matrices for the joints
    R_joints = f.threeD_rotation_set(joint_angles, joint_axes)
    
    # Compute the cumulative product of joint rotations to find link orientations
    R_links = f.rotation_set_cumulative_product(R_joints)
    
    # Rotate link vectors by link orientations
    link_vectors_in_world = f.vector_set_rotate(link_vectors, R_links)
    
    # Compute the cumulative sum of link vectors to get the endpoints
    link_end_set = f.vector_set_cumulative_sum(link_vectors_in_world)
    
    # Add a zero vector for the origin at the base of the first link
    link_end_set_with_base = [np.zeros((3, 1))] + link_end_set
    
    # Convert to a simple matrix
    link_ends = np.hstack(link_end_set_with_base)
    
    return link_ends, R_joints, R_links, link_vectors_in_world, link_end_set, link_end_set_with_base


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

link_ends, R_joints, R_links, link_vectors_in_world, link_end_set, link_end_set_with_base = threeD_robot_arm_endpoints(link_vectors, joint_angles, joint_axes)

print("link_ends:\n", link_ends)
print("\nR_joints:\n", R_joints)
print("\nR_links:\n", R_links)
print("\nlink_vectors_in_world:\n", link_vectors_in_world)
print("\nlink_end_set:\n", link_end_set)
print("\nlink_end_set_with_base:\n", link_end_set_with_base)

