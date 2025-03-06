import numpy as np
from .cache_decorator import cache_function
from .displacement_expansion import DisplacementExpansion
from .midsurface_geometry import MidSurfaceGeometry


@cache_function  # This decorator caches the results of the function to optimize repeated calls with the same inputs.
def displacement_covariant_derivatives(mid_surface_geometry: MidSurfaceGeometry,
                                       displacement_expansion: DisplacementExpansion, i: int, xi1, xi2):
    """
    Computes the covariant derivatives of the displacement field based on the provided
    mid-surface geometry and displacement expansion.

    :param mid_surface_geometry: Instance of MidSurfaceGeometry, used to access geometric quantities like
                                  Christoffel symbols and their derivatives.
    :param displacement_expansion: Instance of DisplacementExpansion, used to compute the displacement and
                                    its derivatives with respect to curvilinear coordinates.
    :param i: The index of the shape function used for displacement computation.
    :param xi1, xi2: The curvilinear coordinates (xi1, xi2) where the derivatives are computed.

    :return: A tuple (dcu, ddcu) where:
             - dcu: The first covariant derivative of the displacement.
             - ddcu: The second covariant derivative of the displacement.
    """

    # C^i_{j alpha}: The Christoffel symbols of the second kind, representing the connection coefficients
    C = mid_surface_geometry.christoffel_symbols(xi1, xi2)

    # C^i_{j alpha}_{,beta}: The first derivative of the Christoffel symbols with respect to the curvilinear coordinates
    dC = mid_surface_geometry.christoffel_symbols_first_derivative(xi1, xi2)

    # u_i: The displacement associated with the shape function i at coordinates (xi1, xi2)
    u = displacement_expansion.shape_function(i, xi1, xi2)

    # u_{i,alpha}: The first derivatives of the displacement u with respect to the curvilinear coordinates
    du = displacement_expansion.shape_function_first_derivatives(i, xi1, xi2)

    # u_{i,alpha beta}: The second derivatives of the displacement u with respect to the curvilinear coordinates
    ddu = displacement_expansion.shape_function_second_derivatives(i, xi1, xi2)

    if displacement_expansion.number_of_fields() == 6:
        v = u[1]
        u = u[0]
        dv = du[1]
        du = du[0]
        ddv = ddu[1]
        ddu = ddu[0]

    # u_{i|alpha}: The covariant derivative of the displacement u_i with respect to xi_alpha.
    # The formula used here subtracts the Christoffel symbols contribution to the derivative.
    dcu = du - np.einsum('jia..., j...->ia...', C, u)

    # u_{i|alpha beta}: The second covariant derivative of the displacement u_i with respect to xi_alpha and xi_beta.
    # The formula accounts for several terms involving the Christoffel symbols and their derivatives.
    ddcu = ddu - np.einsum('jia..., jb...->iab...', C, du) - np.einsum('jiab..., j...->iab...', dC, u) - np.einsum(
        'jib..., ja...->iab...', C, dcu)

    # Return the covariant derivatives
    if displacement_expansion.number_of_fields() == 3:
        return dcu, ddcu
    else:
        # u_{i|alpha}: The covariant derivative of the displacement u_i with respect to xi_alpha.
        # The formula used here subtracts the Christoffel symbols contribution to the derivative.
        dcv = dv - np.einsum('jia..., j...->ia...', C, v)

        # u_{i|alpha beta}: The second covariant derivative of the displacement u_i with respect to xi_alpha and xi_beta.
        # The formula accounts for several terms involving the Christoffel symbols and their derivatives.
        ddcv = ddv - np.einsum('jia..., jb...->iab...', C, dv) - np.einsum('jiab..., j...->iab...', dC, v) - np.einsum(
            'jib..., ja...->iab...', C, dcv)

        return dcu, ddcu, dcv, ddcv
