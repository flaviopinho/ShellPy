import numpy as np

from shellpy import cache_function
from shellpy import displacement_covariant_derivatives
from shellpy import DisplacementExpansion
from shellpy import MidSurfaceGeometry


def koiter_nonlinear_strain_components_quadratic(mid_surface_geometry: MidSurfaceGeometry,
                                                 displacement_expansion: DisplacementExpansion,
                                                 i: int, j: int, xi1, xi2):
    # Calculate the nonlinear components of the Koiter strain tensor

    # Compute the displacement covariant derivatives for two different degrees of freedom (i and j)
    # dcu1, ddcu1: displacement and second derivatives for displacement DOF i
    # dcu2, ddcu2: displacement and second derivatives for displacement DOF j
    dcu1, ddcu1 = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, i, xi1, xi2)
    dcu2, ddcu2 = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, j, xi1, xi2)

    # Get the contravariant components of the metric tensor (G^{alpha beta}) for the mid-surface geometry
    metric_tensor_contravariant_components2 = mid_surface_geometry.metric_tensor_contravariant_components_extended(xi1,
                                                                                                                   xi2)

    # Calculate the nonlinear strain components using the modified metric tensor
    # u1^{p}_{|alpha} = G^{p i} u1_{i|alpha} and u1^{p}_{|alpha} u2_{p|beta}
    gamma_nl = 0.5 * np.einsum('pi...,ia...,pb...->ab...', metric_tensor_contravariant_components2, dcu1, dcu2)

    ###############################################
    rho_nl0 = np.zeros((2, 2) + np.shape(xi1))

    mu1_lin = np.zeros((3,) + np.shape(xi1))
    mu2_lin = np.zeros((3,) + np.shape(xi1))

    mu_nlin = np.zeros((3,) + np.shape(xi1))

    dcu1_contra = np.einsum('mi...,i...->m...', metric_tensor_contravariant_components2, dcu1)
    dcu2_contra = np.einsum('mi...,i...->m...', metric_tensor_contravariant_components2, dcu2)

    mu1_lin[0] = -dcu1_contra[2, 0]
    mu1_lin[1] = -dcu1_contra[2, 1]
    mu1_lin[2] = dcu1_contra[0, 0] + dcu1_contra[1, 1] + np.ones(np.shape(xi1))

    mu2_lin[0] = -dcu2_contra[2, 0]
    mu2_lin[1] = -dcu2_contra[2, 1]
    mu2_lin[2] = dcu2_contra[0, 0] + dcu2_contra[1, 1] + np.ones(np.shape(xi1))

    mu_nlin[0] = dcu1_contra[1, 0] * dcu2_contra[2, 1] - dcu1_contra[2, 0] * dcu2_contra[1, 1]
    mu_nlin[1] = dcu1_contra[2, 0] * dcu2_contra[0, 1] - dcu1_contra[0, 0] * dcu2_contra[2, 1]
    mu_nlin[2] = dcu1_contra[0, 0] * dcu2_contra[1, 1] - dcu1_contra[1, 0] * dcu2_contra[0, 1]

    # Compute the Christoffel symbols C^i_{j alpha} for the given (xi1, xi2) coordinates
    C = mid_surface_geometry.christoffel_symbols(xi1, xi2)

    rho_nl1 = - np.einsum('gab...,g...->ab...', C[:, 0:2], mu_nlin)

    rho_nl3 = -0.5 * np.einsum('ik...,k...,iab...->ab...', metric_tensor_contravariant_components2, mu1_lin, ddcu2)
    rho_nl3 += -0.5 * np.einsum('ik...,k...,iab...->ab...', metric_tensor_contravariant_components2, mu2_lin, ddcu1)

    K = mid_surface_geometry.curvature_tensor_mixed_components(xi1, xi2)
    f1 = np.einsum('sa...,bs...->ab...', K, gamma_nl)
    f2 = np.einsum('sb...,as...->ab...', K, gamma_nl)

    rho_nl4 = - 1 / 2 * (f1 + f2)

    return gamma_nl, rho_nl1 + rho_nl3 + rho_nl4  # Return the nonlinear strain components


def koiter_nonlinear_strain_components_cubic(mid_surface_geometry: MidSurfaceGeometry,
                                             displacement_expansion: DisplacementExpansion,
                                             i: int, j: int, k: int, xi1, xi2):
    # Calculate the nonlinear components of the Koiter strain tensor

    # Compute the displacement covariant derivatives for two different degrees of freedom (i and j)
    # dcu1, ddcu1: displacement and second derivatives for displacement DOF i
    # dcu2, ddcu2: displacement and second derivatives for displacement DOF j
    dcu1, ddcu1 = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, i, xi1, xi2)
    dcu2, ddcu2 = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, j, xi1, xi2)
    dcu3, ddcu3 = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, k, xi1, xi2)

    # Get the contravariant components of the metric tensor (G^{alpha beta}) for the mid-surface geometry
    metric_tensor_contravariant_components2 = mid_surface_geometry.metric_tensor_contravariant_components_extended(xi1,
                                                                                                                   xi2)
    ###############################################
    mu_nlin = np.zeros((3,) + np.shape(xi1))

    dcu1_contra = np.einsum('mi...,i...->m...', metric_tensor_contravariant_components2, dcu1)
    dcu2_contra = np.einsum('mi...,i...->m...', metric_tensor_contravariant_components2, dcu2)

    mu_nlin[0] = dcu1_contra[1, 0] * dcu2_contra[2, 1] - dcu1_contra[2, 0] * dcu2_contra[1, 1]
    mu_nlin[1] = dcu1_contra[2, 0] * dcu2_contra[0, 1] - dcu1_contra[0, 0] * dcu2_contra[2, 1]
    mu_nlin[2] = dcu1_contra[0, 0] * dcu2_contra[1, 1] - dcu1_contra[1, 0] * dcu2_contra[0, 1]

    rho_cubic = - np.einsum('ik...,k...,iab...->ab...', metric_tensor_contravariant_components2, mu_nlin, ddcu3)

    return rho_cubic  # Return the nonlinear strain components
