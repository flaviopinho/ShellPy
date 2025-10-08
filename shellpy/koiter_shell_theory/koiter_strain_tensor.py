import numpy as np

from shellpy import cache_function
from shellpy import displacement_covariant_derivatives
from shellpy import DisplacementExpansion
from shellpy import MidSurfaceGeometry


# @cache_function
def koiter_linear_strain_components(mid_surface_geometry: MidSurfaceGeometry,
                                    displacement_expansion: DisplacementExpansion,
                                    i: int, xi1, xi2):
    # Calculate the linear components of the Koiter strain tensor for the given DOF (i)

    # dcu: displacement covariant derivatives
    # ddcu: second covariant derivatives of displacement
    dcu, ddcu = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, i, xi1, xi2)

    # Extract the third component of the displacement (u3) and its derivatives
    # u3: third displacement component (out-of-plane direction)
    dcu3 = dcu[2]

    # Extract in-plane displacement components (u_alpha) and their covariant derivatives
    dcu = dcu[0:2]

    # Extract the second covariant derivatives for the third displacement component (u3)
    ddcu3 = ddcu[2]

    # Extract in-plane second covariant derivatives
    ddcu = ddcu[0:2]

    # Re-arrange shape of the displacement derivative for matrix transposition
    shape_aux = list(range(dcu.ndim))
    shape_aux[0] = 1  # Swap the first dimension
    shape_aux[1] = 0  # Swap the second dimension

    # Calculate the linear strain components (gamma_{alpha beta}) as the symmetric part of the displacement derivatives
    # gamma_{alpha beta} = 1/2 (u_{alpha|beta} + u_{beta|alpha})
    gamma = 1 / 2 * (dcu + np.transpose(dcu, tuple(shape_aux)))

    # Compute the Christoffel symbols C^i_{j alpha} for the given (xi1, xi2) coordinates
    C = mid_surface_geometry.christoffel_symbols(xi1, xi2)

    # Select the relevant part of the Christoffel symbols for the linear strain calculation
    C = C[0:2, 0:2]

    # Compute the linear strain components rho_{alpha beta} using the Christoffel symbols
    # rho_{alpha beta} = C^gamma_{alpha beta} u_{3| gamma} - u_{3| alpha beta}
    rho = np.einsum('gab...,g...->ab...', C, dcu3) - ddcu3

    K = mid_surface_geometry.curvature_tensor_mixed_components(xi1, xi2)
    f1 = np.einsum('sa...,bs...->ab...', K, gamma)
    f2 = np.einsum('sb...,as...->ab...', K, gamma)

    rho = rho - 0 * (f1+f2)

    return gamma, rho  # Return linear strain (gamma) and (rho)


# @cache_function
def koiter_nonlinear_strain_components_total(mid_surface_geometry: MidSurfaceGeometry,
                                             displacement_expansion: DisplacementExpansion,
                                             i: int, j: int, xi1, xi2):
    # Calculate the nonlinear components of the Koiter strain tensor

    # Compute the displacement covariant derivatives for two different degrees of freedom (i and j)
    # dcu1, ddcu1: displacement and second derivatives for displacement DOF i
    # dcu2, ddcu2: displacement and second derivatives for displacement DOF j
    dcu1, ddcu1 = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, i, xi1, xi2)
    dcu2, ddcu2 = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, j, xi1, xi2)

    # Get the contravariant components of the metric tensor (G^{alpha beta}) for the mid-surface geometry
    metric_tensor_contravariant_components = mid_surface_geometry.metric_tensor_contravariant_components(xi1, xi2)

    # Create a modified version of the metric tensor with dimensions for the 3D space (adding a third component)
    shape = list(np.shape(metric_tensor_contravariant_components))
    shape[0] = 3  # Add the third component for 3D space
    shape[1] = 3
    metric_tensor_contravariant_components2 = np.zeros(shape)
    metric_tensor_contravariant_components2[0:2, 0:2] = metric_tensor_contravariant_components
    metric_tensor_contravariant_components2[2, 2] = 1  # Set the third component to 1 for out-of-plane direction

    # Calculate the nonlinear strain components using the modified metric tensor
    # u1^{p}_{|alpha} = G^{p i} u1_{i|alpha} and u1^{p}_{|alpha} u2_{p|beta}
    gamma_nl = 0.5 * np.einsum('pi...,ia...,pb...->ab...', metric_tensor_contravariant_components2, dcu1, dcu2)

    return gamma_nl # Return the nonlinear strain components


