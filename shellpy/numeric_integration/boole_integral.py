import numpy as np

from shellpy import cache_function
from shellpy.numeric_integration.default_integral_division import n_integral_default_x


@cache_function
def boole_weights_simple_integral(edges, n_x=n_integral_default_x):
    """
    This function calculates the Boole's rule integration weights and integration points
    using the given number of subdivisions.

    :param edges: A tuple object defining the edges of the domain Exemple (-h/2, h/2).
    :param n_x: The number of subdivisions to use (default is 20).
    :return: The integration points (x) and the corresponding weights (W).
    """

    n2_x = n_x * 4  # Total number of subdivisions (4n)

    # Generate the points x (we need 4n + 1 points)
    x = np.linspace(*edges, n2_x + 1)  # Generate x points for the integration domain

    # Apply the weights according to Boole's rule
    weights_base = np.array([14, 32, 12, 32])  # Base pattern of weights for Boole's rule
    weights_1d_x = np.tile(weights_base, n_x)  # Repeat the base pattern n times
    weights_1d_x = np.append(weights_1d_x, 7)  # Add the final weight (7) at the end
    weights_1d_x[0] = 7  # Set the first weight to 7 (adjustment for mid_surface_domain)

    h1 = (edges[1] - edges[0]) / n2_x

    W = (2 * h1 / 45) * weights_1d_x

    return x, W