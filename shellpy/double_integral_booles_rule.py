import numpy as np

from .cache_decorator import cache_function
from .mid_surface_domain import RectangularMidSurfaceDomain

# Default number of divisions for the integral
n_integral_default = 20


def double_integral_booles_rule(func, rectangular_domain: RectangularMidSurfaceDomain, n=n_integral_default):
    """
    This function computes the value of a double integral over a rectangular domain
    using the Boole's rule for numerical integration.

    :param func: The function to be integrated.
    :param rectangular_domain: A RectangularMidSurfaceDomain object defining the edges of the integration domain.
    :param n: The number of subdivisions to use (default is 20).
    :return: The approximate value of the double integral.
    """
    # Get the integration points (x, y) and the weights (W) using Boole's rule
    x, y, W = boole_weights_double_integral(rectangular_domain, n)

    # Evaluate the function at the integration points
    F = func(x, y)

    # Compute the integral as the weighted sum of the function values at the integration points
    integral_value = np.sum(W * F)

    return integral_value


@cache_function
def boole_weights_double_integral(rectangular_boundary: RectangularMidSurfaceDomain, n=n_integral_default):
    """
    This function calculates the Boole's rule integration weights and integration points
    for a rectangular mid_surface_domain using the given number of subdivisions.

    :param rectangular_boundary: A RectangularMidSurfaceDomain object defining the edges of the domain.
    :param n: The number of subdivisions to use (default is 20).
    :return: The integration points (x, y) and the corresponding weights (W).
    """
    n2 = n * 4  # Total number of subdivisions (4n)

    # Generate the points xi1 and xi2 (we need 4n + 1 points)
    xi1 = np.linspace(*rectangular_boundary.edges["xi1"], n2 + 1)  # Generate xi1 points for the integration domain
    xi2 = np.linspace(*rectangular_boundary.edges["xi2"], n2 + 1)  # Generate xi2 points for the integration domain
    x, y = np.meshgrid(xi1, xi2, indexing='xy')  # Create a meshgrid of the xi1 and xi2 points

    # Apply the weights according to Boole's rule
    weights_base = np.array([14, 32, 12, 32])  # Base pattern of weights for Boole's rule
    weights_1d = np.tile(weights_base, n)  # Repeat the base pattern n times
    weights_1d = np.append(weights_1d, 7)  # Add the final weight (7) at the end
    weights_1d[0] = 7  # Set the first weight to 7 (adjustment for mid_surface_domain)

    # Compute the step sizes h1 and h2 for the xi1 and xi2 directions
    h1 = (rectangular_boundary.edges["xi1"][1] - rectangular_boundary.edges["xi1"][0]) / n2
    h2 = (rectangular_boundary.edges["xi2"][1] - rectangular_boundary.edges["xi2"][0]) / n2

    # Create the 2D weight matrix W using the outer product of weights_1d and the step sizes
    W = (2 * 2 * h1 * h2 / (45 * 45)) * np.outer(weights_1d, weights_1d)

    return x, y, W
