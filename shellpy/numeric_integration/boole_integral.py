import numpy as np
from shellpy import cache_function
from shellpy.numeric_integration.default_integral_division import n_integral_default_x

@cache_function
def boole_weights_simple_integral(edges, n_x=n_integral_default_x):
    """
    Calculates Boole's rule integration weights and points for given edges.
    Works for scalar or array edges.

    Parameters
    ----------
    edges : tuple
        A tuple (a, b) defining the integration limits.
        a and b can be scalars or ndarrays of the same shape.
    n_x : int
        Number of Boole's rule segments (default is n_integral_default_x).

    Returns
    -------
    x : ndarray
        Integration points with shape (..., n2_x + 1).
    W : ndarray
        Corresponding integration weights with the same shape.
    """

    a, b = edges
    a = np.asarray(a)
    b = np.asarray(b)

    n2_x = n_x * 4  # Total number of subdivisions (4n)
    shape_h = np.broadcast_shapes(a.shape, b.shape)

    # Normalized positions between 0 and 1 for all integration points
    xi = np.linspace(0, 1, n2_x + 1)

    # Expand to match shape_h
    a_exp = np.expand_dims(a, axis=-1)
    b_exp = np.expand_dims(b, axis=-1)
    xi_exp = xi.reshape((1,) * len(shape_h) + (-1,))

    # Compute integration points
    x = a_exp + (b_exp - a_exp) * xi_exp

    # Base weights according to Boole's rule
    weights_base = np.array([14, 32, 12, 32])
    weights_1d_x = np.tile(weights_base, n_x)
    weights_1d_x = np.append(weights_1d_x, 7)
    weights_1d_x[0] = 7

    # Step size (can be array)
    h1 = (b - a) / n2_x
    h1_exp = np.expand_dims(h1, axis=-1)

    W = (2 * h1_exp / 45) * weights_1d_x

    return x, W