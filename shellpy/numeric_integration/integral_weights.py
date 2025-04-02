import numpy as np
from multipledispatch import dispatch

from shellpy.cache_decorator import cache_function
from shellpy.mid_surface_domain import RectangularMidSurfaceDomain
from .boole_integral import boole_weights_simple_integral
from .default_integral_division import n_integral_default_x, n_integral_default_y


@dispatch(RectangularMidSurfaceDomain, int, int, object)
@cache_function
def double_integral_weights(domain: RectangularMidSurfaceDomain, n_x=n_integral_default_x, n_y=n_integral_default_y,
                            integral_rule=boole_weights_simple_integral):
    """
    Computes the integration points and weights for a double integral over a rectangular mid-surface domain.

    Parameters:
        domain (RectangularMidSurfaceDomain): The domain over which integration is performed.
        n_x (int, optional): Number of integration points in the x-direction. Defaults to n_integral_default_x.
        n_y (int, optional): Number of integration points in the y-direction. Defaults to n_integral_default_y.
        integral_rule (callable, optional): The numerical integration rule. Defaults to boole_weights_simple_integral.

    Returns:
        tuple: Arrays of integration points (xi1, xi2) and corresponding weights (Wxy).
    """
    xi1, Wx = integral_rule(domain.edges["xi1"], n_x)  # Compute integration points and weights for xi1
    xi2, Wy = integral_rule(domain.edges["xi2"], n_y)  # Compute integration points and weights for xi2
    xi1, xi2 = np.meshgrid(xi1, xi2, indexing='ij')  # Create a mesh grid of integration points
    Wxy = np.einsum('i, j->ij', Wx, Wy)  # Compute the weight matrix for double integration
    return xi1, xi2, Wxy


@dispatch(tuple, int, int, object)
@cache_function
def double_integral_weights(edges, n_x=n_integral_default_x, n_y=n_integral_default_y,
                            integral_rule=boole_weights_simple_integral):
    """
    Computes the integration points and weights for a double integral over a general edge-defined domain.

    Parameters:
        edges (tuple): A tuple containing the domain edges for integration.
        n_x (int, optional): Number of integration points in the x-direction. Defaults to n_integral_default_x.
        n_y (int, optional): Number of integration points in the y-direction. Defaults to n_integral_default_y.
        integral_rule (callable, optional): The numerical integration rule. Defaults to boole_weights_simple_integral.

    Returns:
        tuple: Arrays of integration points (xi1, xi2) and corresponding weights (Wxy).
    """
    xi1, Wx = integral_rule(edges[0], n_x)  # Compute integration points and weights for xi1
    xi2, Wy = integral_rule(edges[1], n_y)  # Compute integration points and weights for xi2
    xi1, xi2 = np.meshgrid(xi1, xi2, indexing='ij')  # Create a mesh grid of integration points
    Wxy = np.einsum('i, j->ij', Wx, Wy)  # Compute the weight matrix for double integration
    return xi1, xi2, Wxy