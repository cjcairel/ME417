import numpy as np

def follow_trajectory(t, alpha, J, shape_to_draw):
    # Approximate the derivative of 'shape_to_draw' at time 't' using finite differences

    
    dt=1e-5
    v = (shape_to_draw(t + dt) - shape_to_draw(t)) / dt

    J_matrix = J(alpha)
    alpha_dot, _, _, _ = np.linalg.lstsq(J_matrix, v, rcond=None)

    return alpha_dot, v


#test
shape_to_draw = lambda t: np.array([t, t**2, t**3])
J = lambda a: np.array([[a[0], 0, 0],
                         [0, a[1], 0],
                         [0, 0, 1]])

t = 1
alpha = np.array([2, 3, 0])

alpha_dot, v = follow_trajectory(t, alpha, J, shape_to_draw)
print("alpha_dot:", alpha_dot ,"\n" )
print("v:", v)

