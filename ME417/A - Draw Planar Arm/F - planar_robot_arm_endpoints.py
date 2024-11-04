#Find endpoints of planar arm links


import numpy as np
import Z_Every_Function as F

def planar_robot_arm_endpoints(link_vectors, joint_angles):
    """
    Take a set of link vectors and joint angles, and return a matrix whose
    columns are the endpoints of all of the links (including the point that
    is the first end of the first link, which should be placed at the
    origin).

    Parameters:
    link_vectors: a 1xn cell array, each element of which is a 2x1 vector
                    describing the vector from the base of the corresponding link to
                    its end
    joint_angles: a nx1 vector, each element of which is the joint angle
                    preceeding the corresponding link
    
    Outputs:
    
    link_ends: a 3x(n+1) matrix, whose first column is the location
                of the base of the first link (which should be at the origin), and
                whose remaining columns are the endpoints of the links
    %
    Additional outputs (These are intermediate variables. Having the option
    to return them as outputs lets our automatic code checker tell you
    where problems are in your code):
    %
    R_joints: The rotation matrices associated with the joints
    R_links: The rotation matrices for the link orientations
    link_vectors_in_world: The link vectors in their current orientations
    link_end_set: The endpoints of the links after taking the cumulative
                    sum of link vectors
    """
    
    # Generate the rotation matrices for each joint angle
    R_joints = F.planar_rotation_set(joint_angles)
    
    # Generate the cumulative rotation matrices for link orientations
    R_links = F.rotation_set_cumulative_product(R_joints)
    
    # Rotate each link vector by its corresponding rotation matrix
    link_vectors_in_world = F.vector_set_rotate(link_vectors, R_links)
    
    # Calculate cumulative sum of rotated link vectors to find link endpoints
    link_end_set = F.vector_set_cumulative_sum(link_vectors_in_world)
    
    # Add origin point as the first element in link_end_set_with_base
    origin = np.array([[0], [0]])
    link_end_set_with_base = [origin] + link_end_set
    
    # Convert the list of link vectors with the base origin to a single matrix
    link_ends = np.hstack(link_end_set_with_base)
    
    return link_ends, R_joints, R_links, link_vectors_in_world, link_end_set, link_end_set_with_base



#test code 
link_vectors = [np.array([[1], [0]]), np.array([[1], [0]])]
joint_angles = np.array([np.pi / 4, -np.pi / 2])

link_ends, R_joints, R_links, link_vectors_in_world, link_end_set, link_end_set_with_base = planar_robot_arm_endpoints(link_vectors, joint_angles)


print("Link Ends:\n", link_ends)
print("\nRotation Matrices at Joints (R_joints):")
for i, R in enumerate(R_joints):
    print(f"R_joints[{i}]:\n{R}\n")

print("\nRotation Matrices for Links (R_links):")
for i, R in enumerate(R_links):
    print(f"R_links[{i}]:\n{R}\n")

print("\nLink Vectors in World Frame (link_vectors_in_world):")
for i, vec in enumerate(link_vectors_in_world):
    print(f"link_vectors_in_world[{i}]:\n{vec}\n")

print("\nCumulative Sum of Link Vectors (link_end_set):")
for i, vec in enumerate(link_end_set):
    print(f"link_end_set[{i}]:\n{vec}\n")

print("\nLink Endpoints with Origin (link_end_set_with_base):")
for i, vec in enumerate(link_end_set_with_base):
    print(f"link_end_set_with_base[{i}]:\n{vec}\n")
