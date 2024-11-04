import numpy as np
import matplotlib.pyplot as plt
import Z_Every_Function as F

def ME317_Assignment_draw_3D_arm_with_Jacobian():
    """
    Make a set of plots (subplots in one axis) that illustrate the
    relationship between the geometry of the arm and the Jacobians of the
    links.
    
    Returns:
    --------
    link_vectors, joint_angles, joint_axes, J, link_ends, link_end_set, ax, l, l2, l3, q : various
    """
    # Specify link vectors 
    link_vectors = [np.array([[1], [0], [0]]), np.array([[1], [0], [0]]), np.array([[0], [0], [0.5]])]
    
    # Specify joint angles 
    joint_angles = [2 * np.pi / 5, -np.pi / 4, np.pi / 4]
    
    # Specify joint axes 
    joint_axes = ['z', 'y', 'x']
    
    # Create an empty list for Jacobians, named 'J'
    J = [None] * len(link_vectors)
    
    # Loop over the elements of J to compute Jacobians
    for idx in range(len(J)):
        (J[idx], link_ends, link_end_set, link_end_set_with_base, 
         _, _, joint_axis_vectors_R, _) = F.arm_Jacobian(link_vectors, joint_angles, joint_axes, idx + 1)
    
    # Plotting
    # Create figure and subaxes, store axis handles in a variable named 'ax'
    p = len(link_vectors)
    m = int(np.ceil(np.sqrt(p)))
    n = m
    ax, fig = F.create_subaxes(420, m, n, p)
    
    # Create empty lists 'l', 'l2', 'l3', and 'q' to hold plot handles
    l = [None] * len(J)
    l2 = [None] * len(J)
    l3 = [None] * len(J)
    q = [None] * len(J)
    
    # Loop over subfigure axes to draw robot arm and Jacobians
    for i, axis in enumerate(ax):
        # Draw the robot arm links with line and circles for endpoints
        axis.plot(link_ends[0, :], link_ends[1, :], link_ends[2, :], marker='o')
        
        # Set the axis to 3D view
        axis.set_box_aspect([1, 1, 1])
        axis.view_init(elev=20, azim=120)
        
        # Add arrows for Jacobian columns using `draw_vectors_at_point`
        p = link_end_set[i]
        q[i] = F.draw_vectors_at_point(p, J[i], axis)
        
        # Adding dotted lines to represent joint Jacobians
        l2[i] = [None] * len(J)
        l3[i] = [None] * 3
        for idx in range(i + 1):
            r1 = link_end_set_with_base[idx].flatten()
            axis_vector = joint_axis_vectors_R[idx].flatten()
            unit_axis_vector = axis_vector / np.linalg.norm(axis_vector)
            r2 = link_end_set[i].flatten()
            R2 = r1 + unit_axis_vector
            
            # Plot dotted line from joint to end of current link
            color = q[i][idx].get_color()
            l2[i][idx] = axis.plot([r1[0], r2[0]], [r1[1], r2[1]], [r1[2], r2[2]], color=color, linestyle=":")
            
            # Plot dashed line from joint in direction of joint's axis
            l3[i][idx] = axis.plot([r1[0], R2[0]], [r1[1], R2[1]], [r1[2], R2[2]], color=color, linestyle="--")
    
    plt.show()
    
    return (link_vectors, joint_angles, joint_axes, J, link_ends, link_end_set, ax, l, l2, l3, q)

#test
ME317_Assignment_draw_3D_arm_with_Jacobian()
