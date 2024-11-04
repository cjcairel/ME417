import numpy as np
import matplotlib.pyplot as plt
from Z_Every_Function import planar_robot_arm_endpoints, create_axes

def ME317_Assignment_draw_planar_arm():
    """
    Draw a planar robotic arm as a single line with markers at joint endpoints.

    Returns:
        tuple: Contains the following elements:
            - link_vectors: List of link vectors.
            - joint_angles: Array of joint angles.
            - link_ends: 2D array of link endpoints.
            - ax: The axis handle.
            - line_handle: The handle for the plotted line.
    """
    # Specify link vectors
    link_vectors = [np.array([1, 0]).reshape(2, 1),
                    np.array([1, 0]).reshape(2, 1),
                    np.array([0.5, 0]).reshape(2, 1)]

    # Specify joint angles
    joint_angles = np.array([2 * np.pi / 5, -np.pi / 2, np.pi / 4])

    # Get the endpoints of the links
    link_ends, R_joints, R_links, link_vectors_in_world, link_end_set, link_end_set_with_base = planar_robot_arm_endpoints(link_vectors, joint_angles)

    # Create figure and axes for the plot
    ax, fig = create_axes(42069)

    # Draw a line with markers at the endpoints
    line_handle, = ax.plot(link_ends[0, :], link_ends[1, :], marker='o', linestyle='-')

    return link_vectors, joint_angles, link_ends, ax, line_handle

# test code
link_vectors, joint_angles, link_ends, ax, line_handle = ME317_Assignment_draw_planar_arm()
plt.show()
