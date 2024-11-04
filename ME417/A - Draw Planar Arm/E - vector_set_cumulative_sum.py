#vector_set_cumulative_sum
import numpy as np

def vector_set_cumulative_sum(v_set):
    """
    Take the cumulative sum of a set of vectors.
    
    Input:
    - v_set: list of 2x1 or 3x1 vectors
    
    Output:
    - v_set_s: list of vectors as cumulative sums
    """
    v_set_s = v_set.copy()
    
    # Loop over v_set_s, adding each vector to the next one
    for i in range(1, len(v_set_s)):
        v_set_s[i] += v_set_s[i - 1]
    
    return v_set_s


#test code

v_set = [ np.array([[1],
                    [0]]) ,
         np.array([[0],
                   [1]]) ,
         np.array([[12],
                   [30]]) ]


v_set_s = vector_set_cumulative_sum(v_set)
for i, v in enumerate(v_set_s):
    print(f"Cumulative sum vector {i + 1}:\n{v}\n")


'''
Matlab solution 

ans =

     1
     0


ans =

     1
     1


ans =

    13
    31

this solution

Cumulative sum vector 1:
[[1]
 [0]]

Cumulative sum vector 2:
[[1]
 [1]]

Cumulative sum vector 3:
[[13]
 [31]]

'''