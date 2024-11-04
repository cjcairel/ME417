import numpy as np
import matplotlib.pyplot as plt
import Z_Every_Function as F
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


#test code

link_vectors = [np.array([[1], [0], [0]]), 
                np.array([[0], [1], [0]]), 
                np.array([[1], [0], [1]])]

joint_angles = [np.pi / 4, -np.pi / 2, 1]

joint_axes = ['x', 'y', 'z']

(link_set, R_joints, R_links, link_set_local, 
 link_vectors_in_world, links_in_world, link_end_set, 
 link_end_set_with_base) = F.threeD_robot_arm_links(link_vectors, joint_angles, joint_axes)

ax, f= F.threeD_create_axes(3)  

link_colors = ['r', (0.5, 0.5, 0.5), 'k']

l = threeD_draw_links(link_set, link_colors, ax)

plt.show()