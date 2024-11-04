import numpy as np
import matplotlib.pyplot as plt
import Z_Every_Function as F 

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



# test
link_vectors = [np.array([[1], [0], [0]]), 
                np.array([[1], [0], [0]]), 
                np.array([[0], [0], [1]])]

joint_angles = [np.pi / 4, -np.pi / 2, 1]
joint_axes = ['x', 'y', 'z']

(link_set, R_joints, R_links, link_set_local, 
 link_vectors_in_world, links_in_world, link_end_set, 
 link_end_set_with_base) = F.threeD_robot_arm_links(link_vectors, joint_angles, joint_axes)

fig_num = 3
ax, fig = F.create_axes(fig_num)
ax = fig.add_subplot(111, projection='3d') 
link_colors = ['r', [0.5, 0.5, 0.5], 'k']
l = F.draw_links(link_set, link_colors, ax)
new_joint_angles = [angle + 1 for angle in joint_angles]
(link_set, R_joints, R_links, link_set_local, 
 link_vectors_in_world, links_in_world, link_end_set, 
 link_end_set_with_base) = F.threeD_robot_arm_links(link_vectors, new_joint_angles, joint_axes)   
updated_lines = threeD_update_links(l, link_set)

plt.show()

