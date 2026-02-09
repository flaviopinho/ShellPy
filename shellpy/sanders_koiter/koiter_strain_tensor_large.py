import numpy as np

from ..displacement_covariant_derivative import displacement_first_covariant_derivatives, displacement_second_covariant_derivatives
from ..displacement_expansion import DisplacementExpansion
from ..midsurface_geometry import MidSurfaceGeometry


def koiter_nonlinear_strain_components_quadratic(mid_surface_geometry: MidSurfaceGeometry,
                                                 displacement_expansion: DisplacementExpansion,
                                                 i: int, j: int, xi1, xi2):
    # Calculate the nonlinear components of the Koiter strain tensor

    # Compute the displacement covariant derivatives for two different degrees of freedom (i and j)
    # dcu1, ddcu1: displacement and second derivatives for displacement DOF i
    # dcu2, ddcu2: displacement and second derivatives for displacement DOF j
    u1 = displacement_expansion.shape_function(i, xi1, xi2)
    du1 = displacement_expansion.shape_function_first_derivatives(i, xi1, xi2)
    ddu1 = displacement_expansion.shape_function_second_derivatives(i, xi1, xi2)
    dcu1 = displacement_first_covariant_derivatives(mid_surface_geometry, u1, du1, xi1, xi2)
    ddcu1 = displacement_second_covariant_derivatives(mid_surface_geometry, u1, du1, ddu1, xi1, xi2)

    u2 = displacement_expansion.shape_function(j, xi1, xi2)
    du2 = displacement_expansion.shape_function_first_derivatives(j, xi1, xi2)
    ddu2 = displacement_expansion.shape_function_second_derivatives(j, xi1, xi2)
    dcu2 = displacement_first_covariant_derivatives(mid_surface_geometry, u2, du2, xi1, xi2)
    ddcu2 = displacement_second_covariant_derivatives(mid_surface_geometry, u2, du2, ddu2, xi1, xi2)

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
    u1 = displacement_expansion.shape_function(i, xi1, xi2)
    du1 = displacement_expansion.shape_function_first_derivatives(i, xi1, xi2)
    # ddu1 = displacement_expansion.shape_function_second_derivatives(i, xi1, xi2)
    dcu1 = displacement_first_covariant_derivatives(mid_surface_geometry, u1, du1, xi1, xi2)
    # ddcu1 = displacement_second_covariant_derivatives(mid_surface_geometry, u1, du1, ddu1, i, xi1, xi2)

    u2 = displacement_expansion.shape_function(j, xi1, xi2)
    du2 = displacement_expansion.shape_function_first_derivatives(j, xi1, xi2)
    # ddu2 = displacement_expansion.shape_function_second_derivatives(j, xi1, xi2)
    dcu2 = displacement_first_covariant_derivatives(mid_surface_geometry, u2, du2, xi1, xi2)
    # ddcu2 = displacement_second_covariant_derivatives(mid_surface_geometry, u2, du2, ddu2, i, xi1, xi2)

    u3 = displacement_expansion.shape_function(k, xi1, xi2)
    du3 = displacement_expansion.shape_function_first_derivatives(k, xi1, xi2)
    ddu3 = displacement_expansion.shape_function_second_derivatives(k, xi1, xi2)
    # dcu3 = displacement_first_covariant_derivatives(mid_surface_geometry, u3, du3, xi1, xi2)
    ddcu3 = displacement_second_covariant_derivatives(mid_surface_geometry, u3, du3, ddu3, i, xi1, xi2)

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
