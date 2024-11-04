import numpy as np
import Z_Every_Function as F

def vector_set_rotate(v_set, R_set):
    """
    Rotate a set of vectors specified in local coordinates by a set of
    rotation matrices that specifies the orientations of the frames in which
    the vectors are defined
    
    Inputs:
    - v_set: a 1xn or nx1 cell array, each element of which is a 2x1 or 3x1 vector, 
             which define vectors in local frames of reference

    - R_set: a 1xn or nx1 cell array, each element of which is a 2x2 or 3x3 rotation
             matrix (with the size matching the size of the vectors in v_set),
             which define the orientations of the frames in which the vectors
             from v_set are defined

    Output:
    - v_set_R: a 1xn cell array, each element of which is an 2x1 or 3x1 vector,
             which are the vectors from v_set, but rotated into the world frame

             
    """
    v_set_R = v_set.copy()
    
    # Loop over v_set_R, multiplying each vector by the corresponding rotation matrix
    for i in range(len(v_set_R)):
        v_set_R[i] = np.dot(R_set[i], v_set[i])
    
    return v_set_R

#test code 

v_set = [np.array([[1],
                    [0]]), 
            np.array([[0],
                     [1]]), 
            np.array([[10],
                       [15]])]

R_set = F.planar_rotation_set([-1 * np.pi / 4, 1 * np.pi / 4, 2 * np.pi / 4])

v_set_R = vector_set_rotate(v_set, R_set)

for i, v in enumerate(v_set_R):
    print(f"Rotated vector {i + 1}:\n{v}\n")

'''Rotated vector 1:
[[ 0.70710678]
 [-0.70710678]]

Rotated vector 2:
[[-0.70710678]
 [ 0.70710678]]

Rotated vector 3:
[[-15.]
 [ 10.]]'''