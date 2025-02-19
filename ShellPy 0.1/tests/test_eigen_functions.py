import numpy as np
import matplotlib.pyplot as plt

from expansions import determine_eigenfunctions

"""
The code is designed to test and visualize the eigenfunctions generated by the determine_eigenfunctions function.
"""
if __name__ == "__main__":

    # Number of modes to calculate (number of eigenfunctions to plot)
    n_modos = 10

    # Calling the determine_eigenfunctions function to calculate the eigenfunctions.
    # The input parameters are ('F', 'C') (could represent a specific configuration or type),
    # 1 (likely the first eigenfunction or a mid_surface_domain condition), and n_modos (number of modes to calculate).
    eigen = determine_eigenfunctions(('F', 'C'), 1, n_modos)

    # Create a range of x values for plotting the eigenfunctions (from 0 to 1 with 1000 points)
    x = np.linspace(0, 1, 1000)

    # Loop over each mode and plot the corresponding eigenfunction
    for modo in range(n_modos):
        # Extract the eigenfunction for the current mode (modo) from the eigen dictionary
        # The eigenfunction is a callable (likely a function or lambda) stored in the eigen dictionary.
        func = eigen[(modo, 0)]

        # Evaluate the eigenfunction at the points in x
        y = func(x)

        # Plot the eigenfunction for the current mode
        plt.plot(x, y)

    # Display all the plots (eigenfunctions) on the same figure
    plt.show()