import numpy as np
import matplotlib.pyplot as plt
import Z_Every_Function as F


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



# test code
link_vectors = [np.array([[1], [0]]), 
                np.array([[1], [0]])]

joint_angles = np.array([[np.pi/4], [-np.pi/2]])

(link_set, R_joints, R_links, link_set_local, link_vectors_in_world,
 links_in_world, link_end_set, link_end_set_with_base) = F.planar_robot_arm_links(link_vectors, joint_angles) 
ax, f = F.create_axes(317)

link_colors = ['r', [0.5, 0.5, 0.5]] 

l = draw_links(link_set, link_colors, ax)
plt.show() 
