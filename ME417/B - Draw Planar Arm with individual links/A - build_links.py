import numpy as np

def build_links(link_vectors):
    """
    Take a set of link vectors and augment each with a zero vector representing
    the base of the link.
    
    Parameters:
    -----------
    link_vectors: a 1xn cell array, each element of which is an mx1 vector
                    from the base to end of a link, as seen in its *local* coordinate
                    frame

    Returns:
    --------
    link_set: a 1xn cell array, each element of which is a mx2 matrix whose
                first column is all zeros (representing the base of the link in its
                local frame) and whose second column is the link vector (end of the
                link) in its local frame
    """

    # Create an empty list to hold the augmented link vectors
    link_set = []

    # Loop over each link vector and add a zero column to represent the base of the link
    for vec in link_vectors:
        # Create an mx2 matrix with a column of zeros and the link vector
        link_matrix = np.hstack((np.zeros_like(vec), vec))
        link_set.append(link_matrix)

    return link_set

# Test code
link_vectors = [np.array([[1],
                         [0]]), 
                np.array([[1],
                         [0]]), 
                np.array([[0],
                         [2],
                         [1]])]

link_set = build_links(link_vectors)

for link in link_set:
    print(link)

