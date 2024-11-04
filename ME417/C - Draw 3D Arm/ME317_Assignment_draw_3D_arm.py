import Z_Every_Function as F
import numpy as np
import matplotlib.pyplot as plt


def ME317_Assignment_draw_3D_arm():
    # Specify link vectors 
    link_vectors = [np.array([[1], 
                              [0], 
                              [0]]), 
                    np.array([[1], 
                              [0], 
                              [0]]), 
                    np.array([[0], 
                              [0], 
                              [0.5]])]

    # Specify joint angles 
    joint_angles = [(2 * np.pi / 5), (-np.pi / 4), (np.pi / 4)]

    # Specify joint axes 
    joint_axes = ['z', 'y', 'x']

    # Get the endpoints of the links
    link_ends, R_joints, R_links, link_vectors_in_world, link_end_set, link_end_set_with_base = \
        F.threeD_robot_arm_endpoints(link_vectors, joint_angles, joint_axes)

    # Create figure and axes for the plot
    ax, fig = F.create_axes(5)

    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1.5, 1.5, 1.5])

    # Draw a line from the data, with circles at the endpoints
    l = ax.plot(link_ends[0, :], link_ends[1, :], link_ends[2, :], marker='o')


    plt.show()

    return link_vectors, joint_angles, joint_axes, link_ends, ax, l

ME317_Assignment_draw_3D_arm()