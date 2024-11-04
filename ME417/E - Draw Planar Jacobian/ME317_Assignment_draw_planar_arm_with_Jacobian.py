import numpy as np
import matplotlib.pyplot as plt

import Z_Every_Function as F


def ME317_Assignment_draw_planar_arm_with_Jacobian():
    """
    Make a set of plots (subplots in one axis) that illustrate the relationship between the geometry of the arm and the Jacobians of the links.
    
    Returns:
    Various outputs related to the arm structure and Jacobian plots.
    """
    # Define the link vectors, joint angles, and joint axes
    link_vectors = [np.array([[1], 
                              [0], 
                              [0]]), 
                    np.array([[0.5], 
                              [0], 
                              [0]]), 
                    np.array([[0.5], 
                              [0], 
                              [0]])]
    
    joint_angles = [2 * np.pi / 5, -np.pi / 2, np.pi / 4]
    joint_axes = ['z', 'z', 'z']
    
    # Initialize the Jacobian list
    J = []
    
    for idx in range(len(link_vectors)):
        (Jacobian, link_ends, link_end_set, link_end_set_with_base, 
         v_diff, joint_axis_vectors, joint_axis_vectors_R,
         R_links) = F.arm_Jacobian(link_vectors, joint_angles, joint_axes, idx)
        
        J.append(Jacobian)
    
    # Create subplots for the number of links
    p = len(link_vectors)
    m = int(np.ceil(np.sqrt(p)))
    n = m
    ax, f = F.create_subaxes(316, m, n, p)
    
    # Initialize lists for handles to lines and quivers
    l = []
    l2 = []
    q = []
    
    # Draw the robot arm and the Jacobians in each subplot
    for idx in range(p):
        l.append(ax[idx].plot(link_ends[0, :], link_ends[1, :], link_ends[2, :], marker='o')[0])
        q.append(F.draw_vectors_at_point(link_end_set[idx], J[idx], ax[idx]))
    
    # Draw dotted lines for Jacobian components
    for idx in range(p):
        l2.append([])
        for jdx in range(idx + 1):
            r1 = link_end_set_with_base[jdx]
            r2 = link_end_set[idx]
            X = np.column_stack((r1, r2))
            color = q[idx][jdx].get_color()
            l2[idx].append(ax[idx].plot(X[0, :], X[1, :], X[2, :], color=color, linestyle=':')[0])

    plt.show()
    
    return link_vectors, joint_angles, joint_axes, J, link_ends, link_end_set, ax, l, l2, q

#test
ME317_Assignment_draw_planar_arm_with_Jacobian()