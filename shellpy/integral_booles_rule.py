import numpy as np

from . import ConstantThickness
from .cache_decorator import cache_function
from .mid_surface_domain import RectangularMidSurfaceDomain

# Default number of divisions for the integral
n_integral_default_x = 20
n_integral_default_y = 20
n_integral_default_z = 20


def double_integral_booles_rule(func, edges_x, edges_y, n_x=None, n_y=None):
    """
    This function computes the value of a double integral over a rectangular domain
    using the Boole's rule for numerical integration.

    :param func: The function to be integrated.
    :param edges_x: Interval of x variables. Exemple (0, a).
    :param edges_y: Interval of y variables. Exemple (0, b).
    :param n_x: The number of subdivisions to use (default is 20).
    :param n_y: The number of subdivisions to use (default is 20).
    :return: The approximate value of the double integral.
    """
    if n_x is None:
        n_x = n_integral_default_x
    if n_y is None:
        n_y = n_integral_default_y

    # Get the integration points (x, y) and the weights (W) using Boole's rule
    x, y, W = boole_weights_double_integral(edges_x, edges_y, n_x, n_y)

    # Evaluate the function at the integration points
    F = func(x, y)

    # Compute the integral as the weighted sum of the function values at the integration points
    integral_value = np.sum(W * F)

    return integral_value


@cache_function
def boole_weights_simple_integral(edges, n_x=None):
    """
    This function calculates the Boole's rule integration weights and integration points
    using the given number of subdivisions.

    :param edges: A tuple object defining the edges of the domain Exemple (-h/2, h/2).
    :param n_x: The number of subdivisions to use (default is 20).
    :return: The integration points (x) and the corresponding weights (W).
    """

    if n_x is None:
        n_x = n_integral_default_z

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


@cache_function
def boole_weights_double_integral(edges_x, edges_y, n_x=None, n_y=None):
    """
    This function calculates the Boole's rule integration weights and integration points
    for a rectangular domain using the given number of subdivisions.

    :param edges_x: Interval of x variables. Exemple (0, a).
    :param edges_y: Interval of y variables. Exemple (0, b).
    :param n_x: The number of subdivisions to use (default is 20).
    :param n_y: The number of subdivisions to use (default is 20).
    :return: The integration points (x, y) and the corresponding weights (W).
    """

    if n_x is None:
        n_x = n_integral_default_x
    if n_y is None:
        n_y = n_integral_default_y

    n2_x = n_x * 4  # Total number of subdivisions (4n)
    n2_y = n_y * 4  # Total number of subdivisions (4n)

    # Generate the points xi1 and xi2 (we need 4n + 1 points)
    xi1 = np.linspace(*edges_x, n2_x + 1)  # Generate xi1 points for the integration domain
    xi2 = np.linspace(*edges_y, n2_y + 1)  # Generate xi2 points for the integration domain
    x, y = np.meshgrid(xi1, xi2, indexing='ij')  # Create a meshgrid of the xi1 and xi2 points

    # Apply the weights according to Boole's rule
    weights_base = np.array([14, 32, 12, 32])  # Base pattern of weights for Boole's rule
    weights_1d_x = np.tile(weights_base, n_x)  # Repeat the base pattern n times
    weights_1d_x = np.append(weights_1d_x, 7)  # Add the final weight (7) at the end
    weights_1d_x[0] = 7  # Set the first weight to 7 (adjustment for mid_surface_domain)

    weights_1d_y = np.tile(weights_base, n_y)  # Repeat the base pattern n times
    weights_1d_y = np.append(weights_1d_y, 7)  # Add the final weight (7) at the end
    weights_1d_y[0] = 7  # Set the first weight to 7 (adjustment for mid_surface_domain)

    # Compute the step sizes h1 and h2 for the xi1 and xi2 directions
    h1 = (edges_x[1] - edges_x[0]) / n2_x
    h2 = (edges_y[1] - edges_y[0]) / n2_y

    # Create the 2D weight matrix W using the outer product of weights_1d and the step sizes
    W = (2 * 2 * h1 * h2 / (45 * 45)) * np.einsum('i, j->ij', weights_1d_x, weights_1d_y)

    return x, y, W


@cache_function
def boole_weights_triple_integral(edges_x, edges_y, edges_z, n_x=None, n_y=None, n_z=None):
    """
    This function calculates the Boole's rule integration weights and integration points
    for a rectangular three-dimensional domain using the given number of subdivisions.

    :param edges_x: Interval of x variables. Exemple (0, a).
    :param edges_y: Interval of y variables. Exemple (0, b).
    :param edges_z: Interval of y variables. Exemple (-h/2, h/2).
    :param n_x: The number of subdivisions to use (default is 20).
    :param n_y: The number of subdivisions to use (default is 20).
    :param n_z: The number of subdivisions to use (default is 20).
    :return: The integration points (x, y, z) and the corresponding weights (W).
    """

    if n_x is None:
        n_x = n_integral_default_x
    if n_y is None:
        n_y = n_integral_default_y
    if n_z is None:
        n_z = n_integral_default_z

    n2_x = n_x * 4  # Total number of subdivisions (4n)
    n2_y = n_y * 4  # Total number of subdivisions (4n)
    n2_z = n_z * 4  # Total number of subdivisions (4n)

    # Generate the points xi1 and xi2 (we need 4n + 1 points)
    xi1 = np.linspace(*edges_x, n2_x + 1)  # Generate xi1 points for the integration domain
    xi2 = np.linspace(*edges_y, n2_y + 1)  # Generate xi2 points for the integration domain
    xi3 = np.linspace(*edges_z, n2_z + 1)  # Generate xi3 points for the integration domain
    x, y, z = np.meshgrid(xi1, xi2, xi3, indexing='ij')  # Create a meshgrid of the xi1 and xi2 points

    # Apply the weights according to Boole's rule
    weights_base = np.array([14, 32, 12, 32])  # Base pattern of weights for Boole's rule
    weights_1d_x = np.tile(weights_base, n_x)  # Repeat the base pattern n times
    weights_1d_x = np.append(weights_1d_x, 7)  # Add the final weight (7) at the end
    weights_1d_x[0] = 7  # Set the first weight to 7 (adjustment for mid_surface_domain)

    weights_1d_y = np.tile(weights_base, n_y)  # Repeat the base pattern n times
    weights_1d_y = np.append(weights_1d_y, 7)  # Add the final weight (7) at the end
    weights_1d_y[0] = 7  # Set the first weight to 7 (adjustment for mid_surface_domain)

    weights_1d_z = np.tile(weights_base, n_z)  # Repeat the base pattern n times
    weights_1d_z = np.append(weights_1d_z, 7)  # Add the final weight (7) at the end
    weights_1d_z[0] = 7  # Set the first weight to 7 (adjustment for mid_surface_domain)

    # Compute the step sizes h1 and h2 for the xi1 and xi2 directions
    h1 = (edges_x[1] - edges_x[0]) / n2_x
    h2 = (edges_y[1] - edges_y[0]) / n2_y
    h3 = (edges_z[1] - edges_z[0]) / n2_z

    # Create the 2D weight matrix W using the outer product of weights_1d and the step sizes
    W = (2 * 2 * 2 * h1 * h2 * h3 / (45 * 45 * 45)) * np.einsum('i, j, k->ijk', weights_1d_x, weights_1d_y,
                                                                weights_1d_z)

    return x, y, z, W
