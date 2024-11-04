import numpy as np
import Z_Every_Function as f

def threeD_robot_arm_links(link_vectors, joint_angles, joint_axes):
    """
    Take a set of link vectors, joint angles, and joint axes, and return a
    set of matrices for which the columns of each matrix are the endpoints of
    one of the links.
    
    Parameters:
    -----------
    link_vectors : list of numpy arrays
        Each element is a 3x1 vector describing the vector from the base 
        of the corresponding link to its end.
        
    joint_angles : numpy array
        A nx1 vector, each element of which is the joint angle preceding 
        the corresponding link.
    
    joint_axes : list of strings
        A list representing the axes around which each joint rotates.

    Returns:
    --------
    link_set : list of numpy arrays
        Each element is a 3x2 matrix whose columns are the endpoints of 
        the corresponding link.
    
    R_joints : list of numpy arrays
        The rotation matrices associated with the joints.
    
    R_links : list of numpy arrays
        The rotation matrices for the link orientations.
    
    link_set_local : list of numpy arrays
        The link vectors augmented with a zero vector that represents 
        the start of the link.
    
    link_vectors_in_world : list of numpy arrays
        The link vectors in their current orientations.
    
    links_in_world : list of numpy arrays
        The link start and end points in their current orientations.
    
    link_end_set : list of numpy arrays
        The endpoints of the links after taking the cumulative sum of the 
        link vectors.
    
    link_end_set_with_base : list of numpy arrays
        Contains the origin point at the base of the first link, followed 
        by the endpoints.
    """
    
    # Step 1: Generate rotation matrices for the joints
    R_joints = f.threeD_rotation_set(joint_angles, joint_axes)
    
    # Step 2: Generate rotation matrices for the link orientations
    R_links = f.rotation_set_cumulative_product(R_joints)
    
    # Step 3: Generate the local link endpoint sets
    link_set_local = f.build_links(link_vectors)
    
    # Step 4: Rotate the link vectors into the world frame
    link_vectors_in_world = f.vector_set_rotate(link_vectors, R_links)
    
    # Step 5: Generate link start-and-end matrices in the world frame
    links_in_world = f.build_links(link_vectors_in_world)
    
    # Step 6: Generate the link endpoints by taking the cumulative sum
    link_end_set = f.vector_set_cumulative_sum(link_vectors_in_world)
    
    # Step 7: Add a zero vector for the base of the first link
    link_end_set_with_base = [np.zeros((3, 1))] + link_end_set
    
    # Step 8: Generate the full link sets by adding the basepoints
    link_set = f.place_links(links_in_world, link_end_set_with_base)
    

    return (link_set, R_joints, R_links, link_set_local, 
            link_vectors_in_world, links_in_world, link_end_set, 
            link_end_set_with_base)



#test code
link_vectors = [np.array([[1], 
                          [0], 
                          [0]]), 
                np.array([[0], 
                          [1], 
                          [0]]), 
                np.array([[0], 
                          [0], 
                          [1]])]

joint_angles = [np.pi / 4, -np.pi / 2, np.pi / 4]

joint_axes = ['x', 'y', 'z']

link_set, R_joints,R_links, link_set_local, link_vectors_in_world, links_in_world, link_end_set, link_end_set_with_base = threeD_robot_arm_links(link_vectors, joint_angles, joint_axes)

for link in link_set:
    print(link)
    print()

    '''
[[0. 1.]
 [0. 0.]
 [0. 0.]]

[[1.         1.        ]
 [0.         0.70710678]
 [0.         0.70710678]]

[[1.         0.        ]
 [0.70710678 0.70710678]
 [0.70710678 0.70710678]]
    '''