import matplotlib as plt


def create_subaxes(fignum, m, n, p):
    """
    Clear out a specified figure and create a clean set of axes in that figure with equal-axis aspect ratio.
    
    Parameters:
    fignum (int): The number of the figure (or a figure handle)
    m (int): The number of rows of subplots to create
    n (int): The number of columns of subplots to create
    p (int): The number of subplots to create (should be less than or equal to m*n)
    
    Returns:
    ax (list): A list of handles to the created subplot axes
    f (Figure): A handle to the figure that was created
    """
    f = plt.figure(fignum)
    f.clf() 
    
    ax = []
    for idx in range(p):
        axis = f.add_subplot(m, n, idx + 1, projection='3d')
        axis.set_box_aspect([1, 1, 1]) 
        ax.append(axis)
    
    return ax, f

