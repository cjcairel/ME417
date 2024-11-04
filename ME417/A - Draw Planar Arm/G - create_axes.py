import matplotlib.pyplot as plt

def create_axes(fignum):
    """
    Clear out a specified figure and create a clean set of axes with an equal-axis aspect ratio.

    Parameters:
        fignum (int): The number of the figure or a figure handle in which to create the axes.

    Returns:
        tuple: Contains the following:
            - ax: The created axis handle.
            - fig: The figure handle.
    """
    fig = plt.figure(fignum)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.box = True

    return ax, fig

# test
fignum = 317
ax, fig = create_axes(fignum)
plt.show()
