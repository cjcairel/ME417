import numpy as np
import Z_Every_Function as F

def planar_robot_arm_links(link_vectors, joint_angles):
    """
    Take a set of link vectors and joint angles, and return a set of matrices
    for which the columns of each matrix are the endpoints of one of the links.

    Parameters:
    ----------
    link_vectors : list of numpy.ndarray
        A list containing n elements, each of which is a 2x1 numpy array
        describing the vector from the base of the corresponding link to
        its end.

    joint_angles : numpy.ndarray
        A nx1 numpy array where each element is the joint angle
        preceding the corresponding link.

    Returns:
    -------
    link_set : list of numpy.ndarray
        A list containing n elements, each of which is a 2x2 numpy array,
        whose columns are the endpoints of the corresponding link.

    R_joints : list of numpy.ndarray
        The rotation matrices associated with the joints.

    R_links : list of numpy.ndarray
        The rotation matrices for the link orientations.

    link_set_local : list of numpy.ndarray
        The link vectors augmented with a zero vector that represents the 
        start of the link.

    link_vectors_in_world : list of numpy.ndarray
        The link vectors in their current orientations.

    links_in_world : list of numpy.ndarray
        The link start and end points in their current orientations.

    link_end_set : list of numpy.ndarray
        The endpoints of the links after taking the cumulative
        sum of link vectors.
    """

    # First, generate a list of rotation matrices corresponding to the joint angles
    R_joints = F.planar_rotation_set(joint_angles)

    # Second, generate a list of the orientations of the link frames by taking the cumulative products of the joint rotation matrices
    R_links = F.rotation_set_cumulative_product(R_joints)

    # Third, generate a list of endpoint sets for the links
    link_set_local = F.build_links(link_vectors)

    # Fourth, generate a list of link vectors rotated by the rotation matrices for the links
    link_vectors_in_world = F.vector_set_rotate(link_vectors, R_links)

    # Fifth, generate a list of the link start-and-end matrices rotated by the rotation matrices for the links
    links_in_world = F.build_links(link_vectors_in_world)

    # Sixth, generate a list of endpoints of each link by taking the cumulative sum of the link vectors
    link_end_set = F.vector_set_cumulative_sum(link_vectors_in_world)

    # Seventh, add a zero vector for the origin point at the base of the first link
    link_end_set_with_base = [np.zeros_like(link_end_set[0])] + link_end_set

    # Eighth, generate a list by adding the base point of each link to the start-and-end matrices of that link
    link_set = F.place_links(links_in_world, link_end_set_with_base)

    return link_set, R_joints, R_links, link_set_local, link_vectors_in_world, links_in_world, link_end_set, link_end_set_with_base


#test code
link_vectors = [np.array([[1], [0]]), np.array([[1], [0]]), np.array([[3], [1]]), np.array([[1], [2]])]
joint_angles = np.array([[np.pi / 4], [-np.pi / 2], [np.pi / 3], [1]])

(link_set, R_joints, R_links, link_set_local, link_vectors_in_world,
 links_in_world, link_end_set, link_end_set_with_base) = planar_robot_arm_links(link_vectors, joint_angles)


for i in range(len(link_set)):
    print(f'Link Set {i+1}:\n', link_set[i])

''' compare to ans =

         0    0.7071
         0    0.7071


ans =

    0.7071    1.4142
    0.7071   -0.0000


ans =

    1.4142    4.0532
   -0.0000    1.7424


ans =

    4.0532    2.4520
    1.7424    3.3032
 '''