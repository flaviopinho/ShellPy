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


import jax.numpy as jnp


def total_strain_vector_jax(mid_surface_geometry, u_phys, du_phys, ddu_phys, xi1, xi2):
    """
    Computes the total strain components (membrane and curvature change) for a shell
    based on the Sanders-Koiter theory, including NON-LINEAR curvature terms,
    using JAX for automatic differentiation.

    Parameters:
    -----------
    mid_surface_geometry : MidSurfaceGeometry
        Geometric object containing metric tensors, Christoffel symbols, and curvatures.
    u_phys : jnp.ndarray
        Interpolated physical displacement field [u1, u2, u3].
    du_phys : jnp.ndarray
        First derivatives of the displacement field with respect to xi1, xi2.
    ddu_phys : jnp.ndarray
        Second derivatives of the displacement field with respect to xi1, xi2.
    xi1, xi2 : jnp.ndarray
        Curvilinear coordinates at the integration points.

    Returns:
    --------
    gamma_voigt : jnp.ndarray
        Total Membrane strain vector in Voigt notation [e11, e22, g12].
    rho_voigt : jnp.ndarray
        Total Curvature change vector in Voigt notation [k11, k22, k12].
    """

    def to_voigt_2d_jax(eps):
        """
        Converts a 2x2 tensor to a Voigt notation vector [11, 22, 12].
        Uses engineering shear strain (gamma_12 = eps_12 + eps_21).
        """
        return jnp.stack([
            eps[0, 0],  # e11
            eps[1, 1],  # e22
            (eps[0, 1] + eps[1, 0])  # gamma_12 (Engineering shear strain)
        ], axis=0)

    # --- 1. Geometry Acquisition ---
    G_contra = mid_surface_geometry.metric_tensor_contravariant_components_extended(xi1, xi2)
    K = mid_surface_geometry.curvature_tensor_mixed_components(xi1, xi2)
    C = mid_surface_geometry.christoffel_symbols(xi1, xi2)
    dC = mid_surface_geometry.christoffel_symbols_first_derivative(xi1, xi2)

    # --- 2. Covariant Derivatives ---
    dcu = du_phys - jnp.einsum('jia..., j...->ia...', C, u_phys)
    ddcu = ddu_phys - \
           jnp.einsum('jia..., jb...->iab...', C, du_phys) - \
           jnp.einsum('jiab..., j...->iab...', dC, u_phys) - \
           jnp.einsum('jib..., ja...->iab...', C, dcu)

    # --- 3. Linear Membrane Strain (gamma_lin) ---
    dcu_inplane = dcu[0:2, 0:2]
    gamma_lin = 0.5 * (dcu_inplane + jnp.swapaxes(dcu_inplane, 0, 1))

    # --- 4. Non-linear Membrane Strain (gamma_nl) ---
    # dcu_contra: i_contra, alpha, x, y
    dcu_contra = jnp.einsum('mi..., i... -> m...', G_contra, dcu)
    gamma_nl = 0.5 * jnp.einsum('pa..., pb... -> ab...', dcu_contra, dcu)

    # --- 5. Linear Curvature Change (rho_lin) ---
    dcu3 = dcu[2, :]
    ddcu3 = ddcu[2, :, :]

    rho_base = jnp.einsum('gab..., g... -> ab...', C[0:2, 0:2, 0:2], dcu3) - ddcu3
    f1_lin = jnp.einsum('sa..., bs... -> ab...', K, gamma_lin)
    f2_lin = jnp.einsum('sb..., as... -> ab...', K, gamma_lin)

    rho_lin = rho_base - 0.5 * (f1_lin + f2_lin)

    # --- 6. Non-linear Curvature Change (rho_nl) ---
    # Since we evaluate the total field directly (u1 = u2 = total_u), the cross-terms
    # from the pure Tensorial formulation collapse into direct single-field expressions.

    # Auxiliary vectors 'mu' for the rotation approximation
    mu_lin = jnp.stack([
        -dcu_contra[2, 0],
        -dcu_contra[2, 1],
        dcu_contra[0, 0] + dcu_contra[1, 1] + 1.0
    ], axis=0)

    # Nonlinear part of the rotation vector cross product
    mu_nlin = jnp.stack([
        dcu_contra[1, 0] * dcu_contra[2, 1] - dcu_contra[2, 0] * dcu_contra[1, 1],
        dcu_contra[2, 0] * dcu_contra[0, 1] - dcu_contra[0, 0] * dcu_contra[2, 1],
        dcu_contra[0, 0] * dcu_contra[1, 1] - dcu_contra[1, 0] * dcu_contra[0, 1]
    ], axis=0)

    # 6.1 Christoffel contribution to rotation change
    # Note: C[:, 0:2, 0:2] extracts the full gamma space (3) but only the alpha, beta in-plane space (2x2)
    rho_nl1 = - jnp.einsum('gab..., g... -> ab...', C[:, 0:2, 0:2], mu_nlin)

    # 6.2 Coupling between linear rotation and out-of-plane curvature change
    # In the cross-mode version it was: -0.5 * (term(u1,u2) + term(u2,u1)).
    # Here, since u1 == u2, it perfectly simplifies to just -1.0 * term.
    rho_nl3 = - jnp.einsum('ik..., k..., iab... -> ab...', G_contra, mu_lin, ddcu)

    # 6.3 Sanders-type correction for the nonlinear membrane part
    # We apply the same curvature correction (f1, f2) but using the non-linear membrane strain.
    f1_nl = jnp.einsum('sa..., bs... -> ab...', K, gamma_nl)
    f2_nl = jnp.einsum('sb..., as... -> ab...', K, gamma_nl)
    rho_nl4 = - 0.5 * (f1_nl + f2_nl)

    # Combine non-linear curvature terms
    rho_nl = rho_nl1 + rho_nl3 + rho_nl4

    # --- 7. Final Total Strains ---
    # Returns the combined linear and non-linear strains converted to Voigt notation
    return to_voigt_2d_jax(gamma_lin + gamma_nl), to_voigt_2d_jax(rho_lin + rho_nl)