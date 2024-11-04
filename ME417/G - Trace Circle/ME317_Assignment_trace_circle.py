import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import Z_Every_Function as F

def ME317_Assignment_trace_circle():
    # Define link vectors 
    link_vectors = [np.array([[1], [0], [0]]), 
                    np.array([[1], [0], [0]]), 
                    np.array([[0.75], [0], [0]])]

    # Define joint axes 
    joint_axes = ['z', 'y', 'y']

    # Define the function to trace a circle in the y-z plane
    shape_to_draw = lambda t: F.circle_x(t) * 0.5

    # Jacobian as a separate function
    def compute_jacobian(alpha):
        return F.arm_Jacobian(link_vectors, alpha, joint_axes, len(link_vectors))

    # separate function
    def joint_velocity(t, alpha):
        J_matrix = compute_jacobian(alpha)  # evaluate Jacobian
        v = np.ravel(shape_to_draw(t))  # target velocity
        alpha_dot, _, _, _ = np.linalg.lstsq(J_matrix, v, rcond=None)
        return alpha_dot

    # Set up parameters for the solver
    T = [0, 1]
    a_start = np.array([0, np.pi / 4, -np.pi / 2])

    # Run the solver 
    sol = solve_ivp(joint_velocity, T, a_start, t_eval=np.linspace(T[0], T[1], 100))
    alpha = sol.y

    # Create figure and axes
    ax, fig = F.create_axes(42)

    # Define link colors
    link_colors = ['r', 'g', 'b']

    # Generate the initial link set 
    (link_set, R_joints, R_links, link_set_local, 
     link_vectors_in_world, links_in_world, link_end_set, 
     link_end_set_with_base) = F.threeD_robot_arm_links(link_vectors, a_start, joint_axes)

    # Draw initial links 
    l = F.threeD_draw_links(link_set, link_colors, ax)

    # Initialize path points matrix
    p = np.zeros((3, alpha.shape[1]))

    # Calculate the path points 
    for i in range(alpha.shape[1]):
         
        (link_set, R_joints, R_links, link_set_local, 
         link_vectors_in_world, links_in_world, link_end_set, 
         link_end_set_with_base) = F.threeD_robot_arm_endpoints(link_vectors, alpha[:, i], joint_axes)
        p[:, i] = link_end_set[:, -1]   

    # Plot the path trace
    l_trace, = ax.plot(p[0, :], p[1, :], p[2, :], 'k')

    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect([1, 1, 1])
    fig.canvas.draw()

    # history storage
    link_set_history = []

    # Animate the arm
    for i in range(alpha.shape[1]):
        # Update and capture each component of the link set for the current configuration
        (link_set, R_joints, R_links, link_set_local, 
         link_vectors_in_world, links_in_world, link_end_set, 
         link_end_set_with_base) = F.threeD_robot_arm_links(link_vectors, alpha[:, i], joint_axes)
        
        # Update the plot with new link positions
        F.threeD_update_links(l, link_set)
        
        # Draw and pause to create an animation effect
        plt.draw()
        plt.pause(0.01)
        
        # Store each link configuration in history
        link_set_history.append(link_set)

    return (link_vectors, joint_axes, shape_to_draw, J, joint_velocity, T, a_start, sol, 
            alpha, ax, link_colors, link_set, R_joints, R_links, link_set_local, 
            link_vectors_in_world, links_in_world, link_end_set, 
            link_end_set_with_base, l, p, l_trace, link_set_history)

ME317_Assignment_trace_circle()
