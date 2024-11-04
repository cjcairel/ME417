import numpy as np
import matplotlib.pyplot as plt
import Z_Every_Function as F


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


#test code 

p = np.array([1, 1, 1]).reshape(3, 1)
V = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

ax, f = F.threeD_create_axes(317)

q = draw_vectors_at_point(p, V, ax)

plt.show()