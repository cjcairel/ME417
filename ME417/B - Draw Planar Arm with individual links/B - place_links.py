import numpy as np

def place_links(links_in_world, link_end_set_with_base):
    """
    Use the locations of the ends of a set of links to place the
    start-and-end matrices for the links.

    Parameters:
    -----------
    links_in_world: a 1xn cell array, each element of which is a matrix
                    whose columns are the start-and-end points of a link in its
                    rotated-but-not-translated frame

    link_end_set_with_base: a 1x(n+1) cell array, each element of which is
                            a vector containing the world location of the end of the
                            corresponding link

    Returns:
    --------
    link_set: a 1xn cell array, each element of which is a 2xn matrix
              whose columns are the start-and-end points of the link
              after the link has been placed in the world
    """
    
    # Start by copying links_in_world into a new variable named 'link_set'
    link_set = links_in_world.copy()

    #Loop over link_set, adding each the location of the base of each link to the link's 
    # endpoint matrix to generate the endpoint locations relative to the world origin
    
    for i in range(len(link_set)):
        link_set[i] = link_set[i] + link_end_set_with_base[i]

    return link_set


#test code
links_in_world = [np.array([[0, 0], [0, 1]]), np.array([[0, 2], [0, 1]])]
link_end_set_with_base = [np.array([[0], [0]]), np.array([[0], [1]]), np.array([[2], [2]])]

link_set = place_links(links_in_world, link_end_set_with_base)

for link in link_set:
    print(link)


'''[[0 0]
 [0 1]]
[[0 2]
 [1 2]]
 '''

