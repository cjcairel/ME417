import numpy as np
import matplotlib.pyplot as plt

#move file into folder
import Z_Every_Function as F

def ME317_Assignment_draw_3D_arm_individual_links():
    """
    Draw the arm as a set of lines, one per link.

    """
    
    # Specify link vectors 
    link_vectors = [np.array([[1], [0], [0]]), np.array([[1], [0], [0]]), np.array([[0], [0], [0.5]])]
    

    # Specify joint angles 
    joint_angles = [2*np.pi/5, -np.pi/4, np.pi/4]


    # Specify joint axes 
    joint_axes = ['z', 'y', 'x']
    

    # Specify colors of links
    link_colors = ['k', 'r', 'b']
    

    # Generate link set and other outputs from threeD_robot_arm_links
    link_set, R_joints, R_links, link_set_local, link_vectors_in_world, links_in_world, link_end_set, link_end_set_with_base = F.threeD_robot_arm_links(link_vectors, joint_angles, joint_axes)


    # Generate the joint axis vectors
    joint_axis_vectors = F.threeD_joint_axis_set(joint_axes)


    # Rotate the joint axis vectors by the link orientations
    joint_axis_vectors_R = F.vector_set_rotate(joint_axis_vectors, R_links)


    # Create figure and axes for the plot
    ax, f = F.threeD_create_axes(12)


    # Draw the links and save handles
    l = F.threeD_draw_links(link_set, link_colors, ax)


    # Loop over the joint locations and draw dashed lines for joint axes
    l3 = []
    for idx in range(len(link_set)):
        r1 = link_end_set_with_base[idx]
        Y = np.hstack((r1, r1 + joint_axis_vectors_R[idx].reshape(-1, 1)))
        l3.append(ax.plot(Y[0, :], Y[1, :], Y[2, :], color=link_colors[idx], linestyle='--')[0])


    plt.show()

    return link_vectors, joint_angles, joint_axes, link_colors, link_set, R_links, joint_axis_vectors, joint_axis_vectors_R, ax, l, l3


#test
ME317_Assignment_draw_3D_arm_individual_links()
