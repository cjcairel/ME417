# file containing every function so that they can be used anywhere without pasting them in

#Z at the beginning to make sure its at the bottom of the folder its in - for consistency 

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# DRAW PLANAR ARM 

def R_planar(theta):
    """
    Planar rotation matrix.

    Parameters:
        theta (float): Scalar value for rotation angle in radians.

    Returns:
        array: 2x2 rotation matrix for rotation by theta.
    """
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    R = np.round(R,8)
    return R


def planar_rotation_set(joint_angles):
    """
    Generate a set of planar rotation matrices corresponding to the angles in the input vector.

    Parameters:
        joint_angles: a 1xn or nx1 vector of joint angles

    Returns:
        R_set: a list of planar rotation matrices for each angle in the vector
    """
    
    joint_angles = np.ravel(joint_angles)

    # Create an empty list for rotation matrices
    R_set = []
    
    # Loop over joint angles to create and store each rotation matrix
    for theta in joint_angles:
        R = R_planar(theta)
        R_set.append(R.reshape(2, 2))
        
    return R_set

def rotation_set_cumulative_product(R_set):
    """
    Take the cumulative product of a set of rotation matrices.
    
    Input:
    - R_set: A 1xn or nx1 cell array, each element of which is a 2x2 or 3x3 rotation matrix
    
    Output:
    - R_set_c: A 1xn or nx1 cell array, the ith element of which is a 2x2 or 3x3 rotation
                matrix that is the cumulative product of the rotation matrices in
                R_set from the first matrix up to the ith matrix
    """
    R_set_c = R_set.copy()
    
    # Loop over R_set_c, multiplying each matrix into the one after it
    for i in range(1, len(R_set_c)):
        R_set_c[i] = np.dot(R_set_c[i - 1], R_set_c[i])
    
    return R_set_c

def vector_set_rotate(v_set, R_set):
    """
    Rotate a set of vectors specified in local coordinates by a set of
    rotation matrices that specifies the orientations of the frames in which
    the vectors are defined
    
    Inputs:
    - v_set: a 1xn or nx1 cell array, each element of which is a 2x1 or 3x1 vector, 
             which define vectors in local frames of reference

    - R_set: a 1xn or nx1 cell array, each element of which is a 2x2 or 3x3 rotation
             matrix (with the size matching the size of the vectors in v_set),
             which define the orientations of the frames in which the vectors
             from v_set are defined

    Output:
    - v_set_R: a 1xn cell array, each element of which is an 2x1 or 3x1 vector,
             which are the vectors from v_set, but rotated into the world frame

             
    """
    v_set_R = v_set.copy()
    
    # Loop over v_set_R, multiplying each vector by the corresponding rotation matrix
    for i in range(len(v_set_R)):
        v_set_R[i] = np.dot(R_set[i], v_set[i])
    
    return v_set_R

def vector_set_cumulative_sum(v_set):
    """
    Take the cumulative sum of a set of vectors.
    
    Input:
    - v_set: list of 2x1 or 3x1 vectors
    
    Output:
    - v_set_s: list of vectors as cumulative sums
    """
    v_set_s = v_set.copy()
    
    # Loop over v_set_s, adding each vector to the next one
    for i in range(1, len(v_set_s)):
        v_set_s[i] += v_set_s[i - 1]
    
    return v_set_s

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
    R_joints = planar_rotation_set(joint_angles)
    
    # Generate the cumulative rotation matrices for link orientations
    R_links = rotation_set_cumulative_product(R_joints)
    
    # Rotate each link vector by its corresponding rotation matrix
    link_vectors_in_world = vector_set_rotate(link_vectors, R_links)
    
    # Calculate cumulative sum of rotated link vectors to find link endpoints
    link_end_set = vector_set_cumulative_sum(link_vectors_in_world)
    
    # Add origin point as the first element in link_end_set_with_base
    origin = np.array([[0], [0]])
    link_end_set_with_base = [origin] + link_end_set
    
    # Convert the list of link vectors with the base origin to a single matrix
    link_ends = np.hstack(link_end_set_with_base)
    
    return link_ends, R_joints, R_links, link_vectors_in_world, link_end_set, link_end_set_with_base


def create_axes(fignum):
    """
    Clear out a specified figure and create a clean set of axes with an equal-axis aspect ratio.

    Parameters:
        fignum (int): The number of the figure or a figure handle in which to create the axes.

    Returns:
        tuple: Contains the following:
            - ax: The created axis handle.
            - fig: The figure handle.
    """
    fig = plt.figure(fignum)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.box = True

    return ax, fig

# DRAW PLANAR ARM WITH INDIVIDUAL LINKS

def build_links(link_vectors):
    """
    Take a set of link vectors and augment each with a zero vector representing
    the base of the link.
    
    Parameters:
    -----------
    link_vectors: a 1xn cell array, each element of which is an mx1 vector
                    from the base to end of a link, as seen in its *local* coordinate
                    frame

    Returns:
    --------
    link_set: a 1xn cell array, each element of which is a mx2 matrix whose
                first column is all zeros (representing the base of the link in its
                local frame) and whose second column is the link vector (end of the
                link) in its local frame
    """

    # Create an empty list to hold the augmented link vectors
    link_set = []

    # Loop over each link vector and add a zero column to represent the base of the link
    for vec in link_vectors:
        # Create an mx2 matrix with a column of zeros and the link vector
        link_matrix = np.hstack((np.zeros_like(vec), vec))
        link_set.append(link_matrix)

    return link_set

def place_links(links_in_world, link_end_set_with_base):
    """
    Use the locations of the ends of a set of links to place the
    start-and-end matrices for the links.

    Parameters:
    -----------
    links_in_world: a 1xn cell array, each element of which is a matrix
                    whose columns are the start-and-end points of a link in its
                    rotated-but-not-translated frame

    link_end_set_with_base: a 1x(n+1) cell array, each element of which is
                            a vector containing the world location of the end of the
                            corresponding link

    Returns:
    --------
    link_set: a 1xn cell array, each element of which is a 2xn matrix
              whose columns are the start-and-end points of the link
              after the link has been placed in the world
    """
    
    # Start by copying links_in_world into a new variable named 'link_set'
    link_set = links_in_world.copy()

    #Loop over link_set, adding each the location of the base of each link to the link's 
    # endpoint matrix to generate the endpoint locations relative to the world origin
    
    for i in range(len(link_set)):
        link_set[i] = link_set[i] + link_end_set_with_base[i]

    return link_set


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
    R_joints = planar_rotation_set(joint_angles)

    # Second, generate a list of the orientations of the link frames by taking the cumulative products of the joint rotation matrices
    R_links = rotation_set_cumulative_product(R_joints)

    # Third, generate a list of endpoint sets for the links
    link_set_local = build_links(link_vectors)

    # Fourth, generate a list of link vectors rotated by the rotation matrices for the links
    link_vectors_in_world = vector_set_rotate(link_vectors, R_links)

    # Fifth, generate a list of the link start-and-end matrices rotated by the rotation matrices for the links
    links_in_world = build_links(link_vectors_in_world)

    # Sixth, generate a list of endpoints of each link by taking the cumulative sum of the link vectors
    link_end_set = vector_set_cumulative_sum(link_vectors_in_world)

    # Seventh, add a zero vector for the origin point at the base of the first link
    link_end_set_with_base = [np.zeros_like(link_end_set[0])] + link_end_set

    # Eighth, generate a list by adding the base point of each link to the start-and-end matrices of that link
    link_set = place_links(links_in_world, link_end_set_with_base)

    return link_set, R_joints, R_links, link_set_local, link_vectors_in_world, links_in_world, link_end_set, link_end_set_with_base


def draw_links(link_set, link_colors, ax):
    """
    Draw a set of lines for a link structure into a specified axis.

    Inputs:
    --------
    link_set : list of numpy arrays
        Each entry is a matrix whose columns are the endpoints of the lines
        describing one link of the system (as constructed by
        planar_build_links or planar_build_links_prismatic).
        
    link_colors : list of strings or RGB tuples
        Each entry can be either a standard color string (e.g., 'k' or 'r')
        or a tuple of the RGB values for the color (range from 0 to 1).
        
    ax : matplotlib axis
        The handle to an axis in which to plot the links.

    Output:
    -------
    l : list of Line2D objects
        A list of the same size as link_set, in which each entry is a handle
        to a line structure for that link.
    """

    # Start by creating an empty list of the same size as link_set, named 'l'
    l = [None] * len(link_set)
    
    # Draw a line for each link, with circles at the endpoints, and colors
    # of the lines set to the corresponding element of link_colors, and save
    # the handle for this line in the corresponding element of 'l'
    
    for idx, (link, color) in enumerate(zip(link_set, link_colors)):
        l[idx] = ax.plot(link[0, :], link[1, :], color=color, marker='o')[0]

    return l



#DRAW 3D ARM 

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
    R_joints = threeD_rotation_set(joint_angles, joint_axes)
    
    # Compute the cumulative product of joint rotations to find link orientations
    R_links = rotation_set_cumulative_product(R_joints)
    
    # Rotate link vectors by link orientations
    link_vectors_in_world = vector_set_rotate(link_vectors, R_links)
    
    # Compute the cumulative sum of link vectors to get the endpoints
    link_end_set = vector_set_cumulative_sum(link_vectors_in_world)
    
    # Add a zero vector for the origin at the base of the first link
    link_end_set_with_base = [np.zeros((3, 1))] + link_end_set
    
    # Convert to a simple matrix
    link_ends = np.hstack(link_end_set_with_base)
    
    return link_ends, R_joints, R_links, link_vectors_in_world, link_end_set, link_end_set_with_base
#DRAW 3D ARM WITH INDIVIDUAL LINKS 

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
    R_joints = threeD_rotation_set(joint_angles, joint_axes)
    
    # Step 2: Generate rotation matrices for the link orientations
    R_links = rotation_set_cumulative_product(R_joints)
    
    # Step 3: Generate the local link endpoint sets
    link_set_local = build_links(link_vectors)
    
    # Step 4: Rotate the link vectors into the world frame
    link_vectors_in_world = vector_set_rotate(link_vectors, R_links)
    
    # Step 5: Generate link start-and-end matrices in the world frame
    links_in_world = build_links(link_vectors_in_world)
    
    # Step 6: Generate the link endpoints by taking the cumulative sum
    link_end_set = vector_set_cumulative_sum(link_vectors_in_world)
    
    # Step 7: Add a zero vector for the base of the first link
    link_end_set_with_base = [np.zeros((3, 1))] + link_end_set
    
    # Step 8: Generate the full link sets by adding the basepoints
    link_set = place_links(links_in_world, link_end_set_with_base)
    

    return (link_set, R_joints, R_links, link_set_local, 
            link_vectors_in_world, links_in_world, link_end_set, 
            link_end_set_with_base)


def threeD_draw_links(link_set, link_colors, ax):
    """
    Draw a set of lines for a link structure into a specified axis.

    Parameters:
    -----------
    link_set : A 1xn cell array, each entry of which is a matrix whose
                columns are the endpoints of the lines describing one link 
                of thesystem (as constructed by build_links with threeD input)
        
    link_colors : 1xn cell array. Each entry can be either a standard matlab
                color string (e.g., 'k' or 'r') or a 1x3 vector of the RGB 
                valuesfor the color (range from 0 to 1)
        
    ax : The handle to an axis in which to plot the links

    Returns:
    --------
    l : A cell array of the same size as link_set, in which each entry is a
        handle to a line structure for that link
    """
    
    # Step 1: Create an empty list for line handles
    l = [None] * len(link_set)
    
    # Step 2: Draw a line for each link, with circles at the endpoints, and store the handle in 'l'
    for idx, (link, color) in enumerate(zip(link_set, link_colors)):
        l[idx] = ax.plot(link[0, :], link[1, :], link[2, :], color=color, marker='o')[0]

    
    return l

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

#DRAW PLANAR JACOBIAN

def vector_set_difference(v, v_set):
    """
    Find the vector difference v - v_set (the vector pointing to v from each element of v_set).
    
    Parameters:
    v (ndarray): A vector
    v_set (list): A list of vectors, each of which is the same size as v
    
    Returns:
    v_diff (list): A list of vectors, each of which is the difference between v and the corresponding vector in v_set
    """
    #Start by copying v_set into a new variable v_diff
    v_diff = v_set.copy()
    #Loop over v_diff, subtracting each vector from v
    for i in range(len(v_diff)):
        v_diff[i] = v - v_diff[i]
    return v_diff

def draw_vectors_at_point(p, V, ax):
    """
    Draw the columns of V as arrows based at point p, in axis ax.

    Parameters:
    p (ndarray): A 3x1 vector designating the location of the vectors.
    V (ndarray): A 3xn matrix, each column of which is a 3x1 vector that should be drawn at point p.
    ax (matplotlib axis): The axis in which to draw the vectors.

    Returns:
    q (list): A list of handles to the quiver objects for the drawn arrows.
    """
    # Create an empty list for storing handles
    q = [None] * V.shape[1]
    
    # Loop over the columns of V
    for idx in range(V.shape[1]):
        # Plot the arrow using quiver and store the handle in the list
        q[idx] = ax.quiver(p[0], p[1], p[2], V[0, idx], V[1, idx], V[2, idx])
    
    return q


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
      link_end_set, link_end_set_with_base) = threeD_robot_arm_endpoints(link_vectors, joint_angles, joint_axes)
    
    # Step 2: Vector difference from each link endpoint to the end of the specified link
    v_diff = vector_set_difference(link_end_set_with_base[link_number], link_end_set_with_base)
    
    # Step 3: Generate joint axis vectors in local coordinates
    joint_axis_vectors = threeD_joint_axis_set(joint_axes)
    
    # Step 4: Rotate joint axes into world coordinates
    joint_axis_vectors_R = vector_set_rotate(joint_axis_vectors, R_links)
    
    # Step 5: Initialize Jacobian as a zero matrix
    J = np.zeros((3, len(joint_angles)), dtype=object)
    
    # Step 6: Fill in the Jacobian matrix up to the specified link
    for i in range(link_number):
        J[:, i] = np.cross(joint_axis_vectors_R[i].flatten(), v_diff[i].flatten())
    
    return J, link_ends, link_end_set, link_end_set_with_base, v_diff, joint_axis_vectors, joint_axis_vectors_R, R_links


def create_subaxes(fignum, m, n, p):
    """
    Clear out a specified figure and create a clean set of axes in that figure with equal-axis aspect ratio.
    
    Parameters:
    fignum (int): The number of the figure (or a figure handle)
    m (int): The number of rows of subplots to create
    n (int): The number of columns of subplots to create
    p (int): The number of subplots to create (should be less than or equal to m*n)
    
    Returns:
    ax (list): A list of handles to the created subplot axes
    f (Figure): A handle to the figure that was created
    """
    f = plt.figure(fignum)
    f.clf() 
    
    ax = []
    for idx in range(p):
        axis = f.add_subplot(m, n, idx + 1, projection='3d')
        axis.set_box_aspect([1, 1, 1]) 
        ax.append(axis)

#TRACE CIRLCE

def threeD_update_links(l, link_set):
    """
    Update the drawings of a set of lines for a link structure

    Inputs:
        l: A list of the same size as link_set, in which each entry is a
           handle to a line structure for that link

        link_set: A list of n elements, each entry of which is a numpy array
           whose columns are the endpoints of the lines describing one link of the
           system (as constructed by build_links)

    Output:
        l: A list of the same size as link_set, in which each entry is a
           handle to a surface structure for that link
    """

    # Loop over the lines whose handles are in 'l', replacing their 'XData',
    # 'YData', and 'ZData' with the information from 'link_set'
    for i in range(len(l)):
        line_handle = l[i]
        link_data = link_set[i]

        XData = link_data[0, :] 
        YData = link_data[1, :] 
        ZData = link_data[2, :] 

        line_handle.set_xdata(XData)
        line_handle.set_ydata(YData)
        line_handle.set_3d_properties(ZData)

        l[i] = line_handle

    return l


def follow_trajectory(t, alpha, J, shape_to_draw):
    # Approximate the derivative of 'shape_to_draw' at time 't' using finite differences

    dt=1e-5
    v = (shape_to_draw(t + dt) - shape_to_draw(t)) / dt

    J_matrix = J(alpha)
    alpha_dot, _, _, _ = np.linalg.lstsq(J_matrix, v, rcond=None)

    return alpha_dot, v


def circle_x(t):
    # Generate points on a unit circle in the y-z plane, wrapping in the 
    # clockwise (negative) direction around the x axis, such that the
    # range of t=[0 1] corresponds to one full cycle of the circle, and
    # the initial point is at [0;0;1]

    # First, make t a 1xn vector if it isn't already one. Use 't.flatten()' to
    # turn t into a row vector
    t = np.array(t).flatten()

    # Second, use 'np.zeros', 'np.sin', and 'np.cos' operations to convert t into
    # xyz points. Don't forget to use a 2*np.pi factor so that the period of
    # the circle is 1. Save the output as variable 'xyz'
    xyz = np.array([np.zeros(t.shape), 
                    np.round(np.sin(2 * np.pi * t),8), 
                    np.round(np.cos(2 * np.pi * t),8)])
    
    return xyz