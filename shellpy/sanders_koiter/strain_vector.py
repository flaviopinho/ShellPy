import numpy as np

from ..displacement_covariant_derivative import displacement_first_covariant_derivatives, \
    displacement_second_covariant_derivatives
from ..displacement_expansion import DisplacementExpansion
from ..midsurface_geometry import MidSurfaceGeometry


def to_voigt_2d(eps):
    """
    Converts a 2x2 strain tensor to a 3-element Voigt notation vector.
    Order: [e11, e22, gamma12]
    Note: gamma12 is the engineering shear strain (eps12 + eps21).
    """
    return np.stack([
        eps[0, 0],  # Normal strain xi1
        eps[1, 1],  # Normal strain xi2
        (eps[0, 1] + eps[1, 0])  # Engineering shear strain
    ], axis=0)


def linear_sanders_koiter_strain_vector(mid_surface_geometry: MidSurfaceGeometry,
                                        displacement_expansion: DisplacementExpansion,
                                        i: int, xi1, xi2):
    """
    Calculates the linear components of the membrane strain (epsilon0) and
    curvature change (epsilon1) for the i-th degree of freedom (DOF).

    This follows the Sanders-Koiter shell theory, which ensures zero strain
    under rigid body motions.
    """
    # --- 1. Shape Functions and Covariant Derivatives ---
    # Get modal shapes and their local derivatives for DOF 'i'
    u = displacement_expansion.shape_function(i, xi1, xi2)
    du = displacement_expansion.shape_function_first_derivatives(i, xi1, xi2)
    ddu = displacement_expansion.shape_function_second_derivatives(i, xi1, xi2)

    # Compute first and second covariant derivatives (u_{i|alpha} and u_{i|alpha beta})
    dcu = displacement_first_covariant_derivatives(mid_surface_geometry, u, du, xi1, xi2)
    ddcu = displacement_second_covariant_derivatives(mid_surface_geometry, u, du, ddu, xi1, xi2)

    # Component extraction: u3 (out-of-plane) and u_alpha (in-plane components 1, 2)
    dcu3 = dcu[2]
    dcu_inplane = dcu[0:2]
    ddcu3 = ddcu[2]

    # --- 2. Membrane Linear Strain (epsilon0 / gamma_alpha_beta) ---
    # Symmetric part of the in-plane covariant derivative
    aux = np.swapaxes(dcu_inplane, 0, 1)
    gamma = 0.5 * (dcu_inplane + aux)

    # --- 3. Curvature Change (epsilon1 / rho_alpha_beta) ---
    # Retrieve Christoffel symbols and mixed curvature tensor components
    C = mid_surface_geometry.christoffel_symbols(xi1, xi2)
    C_reduced = C[0:2, 0:2]  # Surface-only connections

    # Basic curvature change: rho = C^gamma_{alpha beta} * u3|gamma - u3|alpha,beta
    rho = np.einsum('gab...,g...->ab...', C_reduced, dcu3) - ddcu3

    # Sanders Correction: Couples curvature with membrane strain to maintain
    # invariance under rigid body rotation.
    K = mid_surface_geometry.curvature_tensor_mixed_components(xi1, xi2)
    f1 = np.einsum('sa...,bs...->ab...', K, gamma)
    f2 = np.einsum('sb...,as...->ab...', K, gamma)

    rho = rho - 0.5 * (f1 + f2)

    # --- 4. Final Output in Voigt Notation ---
    epsilon0 = to_voigt_2d(gamma)
    epsilon1 = to_voigt_2d(rho)

    return epsilon0, epsilon1


def nonlinear_koiter_strain_components_quadratic_vector(mid_surface_geometry: MidSurfaceGeometry,
                                                        displacement_expansion: DisplacementExpansion,
                                                        i: int, j: int, xi1, xi2):
    """
    Computes the quadratic (nonlinear) components of the Koiter strain tensor
    for the pair of DOFs (i, j).

    In the tensorial Koiter approach, these components form the foundation
    of the U3 and U4 energy tensors.
    """
    # --- 1. Basis Functions for both DOFs ---
    # Field 'i'
    u1 = displacement_expansion.shape_function(i, xi1, xi2)
    du1 = displacement_expansion.shape_function_first_derivatives(i, xi1, xi2)
    ddu1 = displacement_expansion.shape_function_second_derivatives(i, xi1, xi2)
    dcu1 = displacement_first_covariant_derivatives(mid_surface_geometry, u1, du1, xi1, xi2)
    ddcu1 = displacement_second_covariant_derivatives(mid_surface_geometry, u1, du1, ddu1, xi1, xi2)

    # Field 'j'
    u2 = displacement_expansion.shape_function(j, xi1, xi2)
    du2 = displacement_expansion.shape_function_first_derivatives(j, xi1, xi2)
    ddu2 = displacement_expansion.shape_function_second_derivatives(j, xi1, xi2)
    dcu2 = displacement_first_covariant_derivatives(mid_surface_geometry, u2, du2, xi1, xi2)
    ddcu2 = displacement_second_covariant_derivatives(mid_surface_geometry, u2, du2, ddu2, xi1, xi2)

    # --- 2. Geometric Metrics ---
    # Extended contravariant metric tensor G^{pi} used for indices contraction
    G_contra = mid_surface_geometry.metric_tensor_contravariant_components_extended(xi1, xi2)

    # --- 3. Nonlinear Membrane Components (gamma_nl) ---
    # Calculated as: 0.5 * G^{pi} * u_i|a * u_p|b
    # This represents the Green-Lagrange strain contribution for large rotations.
    gamma_nl = 0.5 * np.einsum('pi...,ia...,pb...->ab...', G_contra, dcu1, dcu2)

    # --- 4. Nonlinear Curvature Components (rho_nl) ---
    # Raising indices of the first covariant derivatives
    dcu1_contra = np.einsum('mi...,i...->m...', G_contra, dcu1)
    dcu2_contra = np.einsum('mi...,i...->m...', G_contra, dcu2)

    # Auxiliary vectors 'mu' for the rotation approximation
    mu1_lin = np.array([-dcu1_contra[2, 0],
                        -dcu1_contra[2, 1],
                        dcu1_contra[0, 0] + dcu1_contra[1, 1] + 1.0])

    mu2_lin = np.array([-dcu2_contra[2, 0],
                        -dcu2_contra[2, 1],
                        dcu2_contra[0, 0] + dcu2_contra[1, 1] + 1.0])

    # Nonlinear part of the rotation vector cross product
    mu_nlin = np.array([
        dcu1_contra[1, 0] * dcu2_contra[2, 1] - dcu1_contra[2, 0] * dcu2_contra[1, 1],
        dcu1_contra[2, 0] * dcu2_contra[0, 1] - dcu1_contra[0, 0] * dcu2_contra[2, 1],
        dcu1_contra[0, 0] * dcu2_contra[1, 1] - dcu1_contra[1, 0] * dcu2_contra[0, 1]
    ])

    C = mid_surface_geometry.christoffel_symbols(xi1, xi2)
    K = mid_surface_geometry.curvature_tensor_mixed_components(xi1, xi2)

    # Combining nonlinear rho parcels:
    # 1. Christoffel contribution to rotation change
    rho_nl1 = - np.einsum('gab...,g...->ab...', C[:, 0:2], mu_nlin)

    # 2. Coupling between linear rotation and out-of-plane curvature change
    rho_nl3 = -0.5 * (np.einsum('ik...,k...,iab...->ab...', G_contra, mu1_lin, ddcu2) +
                      np.einsum('ik...,k...,iab...->ab...', G_contra, mu2_lin, ddcu1))

    # 3. Sanders-type correction for the nonlinear membrane part
    f1 = np.einsum('sa...,bs...->ab...', K, gamma_nl)
    f2 = np.einsum('sb...,as...->ab...', K, gamma_nl)
    rho_nl4 = - 0.5 * (f1 + f2)

    rho_nl = rho_nl1 + rho_nl3 + rho_nl4

    # --- 5. Return in Voigt Notation ---
    return to_voigt_2d(gamma_nl), to_voigt_2d(rho_nl)