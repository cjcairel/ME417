import numpy as np

def vector_set_difference(v, v_set):
    """
    Find the vector difference v - v_set (the vector pointing to v from each element of v_set).
    
    Parameters:
    v (ndarray): A vector
    v_set (list): A list of vectors, each of which is the same size as v
    
    Returns:
    v_diff (list): A list of vectors, each of which is the difference between v and the corresponding vector in v_set
    """
    #Start by copying v_set into a new variable v_diff
    v_diff = v_set.copy()
    #Loop over v_diff, subtracting each vector from v
    for i in range(len(v_diff)):
        v_diff[i] = v - v_diff[i]
    return v_diff

#test code

v_set = [np.array([[1], [0]]), np.array([[0], [1]])]

v_single = np.array([[2], [2]])

v_diff = vector_set_difference(v_single, v_set)

for vec in v_diff:
    print(vec)

