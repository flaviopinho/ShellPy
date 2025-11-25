import numpy as np
from .cache_decorator import cache_function
from .displacement_expansion import DisplacementExpansion
from .midsurface_geometry import MidSurfaceGeometry


@cache_function  # This decorator caches the results of the function to optimize repeated calls with the same inputs.
def displacement_first_covariant_derivatives(mid_surface_geometry: MidSurfaceGeometry,
                                             u, du, xi1, xi2):
    """
    Computes the first covariant derivatives of the displacement field based on the provided
    mid-surface geometry and displacement expansion.

    :param mid_surface_geometry: Instance of MidSurfaceGeometry, used to access geometric quantities like
                                  Christoffel symbols and their derivatives.
    :param u: The displacement at coordinates (xi1, xi2)
    :param du: The displacement first derivatives at coordinates (xi1, xi2)
    :param ddu: The displacement second derivatives at coordinates (xi1, xi2)
    :param xi1, xi2: The curvilinear coordinates (xi1, xi2) where the derivatives are computed.

    :return: The first covariant derivative of the displacement.
    """

    # C^i_{j alpha}: The Christoffel symbols of the second kind, representing the connection coefficients
    C = mid_surface_geometry.christoffel_symbols(xi1, xi2)

    # u_{i|alpha}: The covariant derivative of the displacement u_i with respect to xi_alpha.
    # The formula used here subtracts the Christoffel symbols contribution to the derivative.
    dcu = du - np.einsum('jia..., j...->ia...', C, u)

    return dcu


@cache_function  # This decorator caches the results of the function to optimize repeated calls with the same inputs.
def displacement_second_covariant_derivatives(mid_surface_geometry: MidSurfaceGeometry,
                                              u, du, ddu, xi1, xi2):
    """
    Computes the second covariant derivatives of the displacement field based on the provided
    mid-surface geometry and displacement expansion.

    :param mid_surface_geometry: Instance of MidSurfaceGeometry, used to access geometric quantities like
                                  Christoffel symbols and their derivatives.
    :param u: The displacement at coordinates (xi1, xi2)
    :param du: The displacement first derivatives at coordinates (xi1, xi2)
    :param ddu: The displacement second derivatives at coordinates (xi1, xi2)
    :param xi1, xi2: The curvilinear coordinates (xi1, xi2) where the derivatives are computed.

    :return: The second covariant derivative of the displacement.
    """

    # C^i_{j alpha}: The Christoffel symbols of the second kind, representing the connection coefficients
    C = mid_surface_geometry.christoffel_symbols(xi1, xi2)

    # C^i_{j alpha}_{,beta}: The first derivative of the Christoffel symbols with respect to the curvilinear coordinates
    dC = mid_surface_geometry.christoffel_symbols_first_derivative(xi1, xi2)

    # u_{i|alpha}: The covariant derivative of the displacement u_i with respect to xi_alpha.
    # The formula used here subtracts the Christoffel symbols contribution to the derivative.
    dcu = du - np.einsum('jia..., j...->ia...', C, u)

    # u_{i|alpha beta}: The second covariant derivative of the displacement u_i with respect to xi_alpha and xi_beta.
    # The formula accounts for several terms involving the Christoffel symbols and their derivatives.
    ddcu = ddu - np.einsum('jia..., jb...->iab...', C, du) - np.einsum('jiab..., j...->iab...', dC, u) - np.einsum(
        'jib..., ja...->iab...', C, dcu)

    return ddcu
