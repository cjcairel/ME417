import numpy as np

def circle_x(t):
    # Generate points on a unit circle in the y-z plane, wrapping in the 
    # clockwise (negative) direction around the x axis, such that the
    # range of t=[0 1] corresponds to one full cycle of the circle, and
    # the initial point is at [0;0;1]

    # First, make t a 1xn vector if it isn't already one. Use 't.flatten()' to
    # turn t into a row vector
    t = np.array(t).flatten()

    # Second, use 'np.zeros', 'np.sin', and 'np.cos' operations to convert t into
    # xyz points. Don't forget to use a 2*np.pi factor so that the period of
    # the circle is 1. Save the output as variable 'xyz'
    xyz = np.array([np.zeros(t.shape), 
                    np.round(np.sin(2 * np.pi * t),8), 
                    np.round(np.cos(2 * np.pi * t),8)])
    
    return xyz

#test code

# Column input
t_column = np.array([[0.125], [0.25], [0.375], [0.5]])
xyz_from_column = circle_x(t_column)

# Row input
t_row = np.array([0.625, 0.75, 0.875, 1])
xyz_from_row = circle_x(t_row)

print(xyz_from_column,"\n")
print(xyz_from_row)


'''[[ 0.          0.          0.          0.        ]
 [ 0.70710678  1.          0.70710678  0.        ]
 [ 0.70710678  0.         -0.70710678 -1.        ]]

[[ 0.          0.          0.          0.        ]
 [-0.70710678 -1.         -0.70710678 -0.        ]
 [-0.70710678 -0.          0.70710678  1.        ]]'''