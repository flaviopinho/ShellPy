import jax
import jax.numpy as jnp
import numpy as np

from ..fsdt6.shear_correction_factor import shear_correction_factor
from ..shell import Shell
from ..fsdt6.constitutive_matrix_in_shell_frame import constitutive_matrix_in_shell_frame
from ..fsdt7_eas.constitutive_matriz_in_material_frame import constitutive_matrix_in_material_frame
from ..fsdt7_eas.enhanced_assumed_strain import enhanced_assumed_strain_full, enhanced_assumed_strain
from .jax_strain_vector import fsdt7_total_strain_vector_jax
from ..numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_y, \
    n_integral_default_z
from ..numeric_integration.gauss_integral import gauss_weights_simple_integral
from ..numeric_integration.integral_weights import double_integral_weights

# Enforce 64-bit precision in JAX to prevent numerical instability in finite elements
jax.config.update("jax_enable_x64", True)


def fsdt7_strain_energy_internal_force_tangent_matrix_jax(shell: Shell, eas_field,
                                                          n_x=n_integral_default_x,
                                                          n_y=n_integral_default_y,
                                                          n_z=n_integral_default_z,
                                                          integral_method=gauss_weights_simple_integral):
    # ====================================================================
    # --- 1. PRE-PROCESSING (NumPy - Executed once on CPU) ---
    # ====================================================================

    # Generate integration points and weights for the mid-surface and thickness
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)
    h = shell.thickness(xi1, xi2)
    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    n_xy = np.shape(xi1)
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    n_alpha = eas_field.number_of_degrees_of_freedom()  # EAS internal degrees of freedom

    # Geometry mapping variables
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)
    Wxy1 = sqrtG * Wxy  # Combined mid-surface integration weights

    # --- Through-thickness integration of the Constitutive Matrices ---
    C_material = np.copy(
        constitutive_matrix_in_material_frame(shell.mid_surface_geometry, shell.material, (xi1, xi2, xi3)))

    # Broadcast to 3D integration points if it's a 2D matrix
    if C_material.ndim == 2:
        C_material = np.einsum('ij, xyz->ijxyz', C_material, xi3 ** 0)

    # Apply shear correction factors to transverse shear terms
    kappa_x, kappa_y, kappa_xy = shear_correction_factor(C_material, xi3, Wz, det_shifter_tensor)
    C_material[4, 4] = np.einsum('xyz, xy->xyz', C_material[4, 4], kappa_x)
    C_material[3, 3] = np.einsum('xyz, xy->xyz', C_material[3, 3], kappa_y)
    C_material[3, 4] = np.einsum('xyz, xy->xyz', C_material[3, 4], kappa_xy)
    C_material[4, 3] = np.einsum('xyz, xy->xyz', C_material[4, 3], kappa_xy)

    # Transform from material frame to local shell frame
    C = constitutive_matrix_in_shell_frame(shell.mid_surface_geometry, C_material, (xi1, xi2, xi3))

    # Calculate thickness-integrated stiffness matrices (C0 through C4)
    # This pre-integrates the z-coordinate out of the equations
    detWz = det_shifter_tensor * Wz
    C0 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 0 * detWz, optimize=True)
    C1 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 1 * detWz, optimize=True)
    C2 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 2 * detWz, optimize=True)
    C3 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 3 * detWz, optimize=True)
    C4 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 4 * detWz, optimize=True)

    # --- Evaluate Shape Functions ---
    # Phi handles primary displacements (u, v stacked -> 6 components)
    Phi = np.zeros((n_dof, 6) + n_xy)
    dPhi = np.zeros((n_dof, 6, 2) + n_xy)

    for n in range(n_dof):
        Phi[n] = shell.displacement_expansion.shape_function(n, xi1, xi2)
        dPhi[n] = shell.displacement_expansion.shape_function_first_derivatives(n, xi1, xi2)

    # Evaluate Enhanced Assumed Strain (EAS) shape functions
    Phi_alpha = enhanced_assumed_strain_full(eas_field, (xi1, xi2, xi3), Wxy1, detWz)

    # ====================================================================
    # --- 2. JAX CONVERSION (Pushing arrays to JAX memory) ---
    # ====================================================================
    j_Phi, j_dPhi = jnp.array(Phi), jnp.array(dPhi)
    j_Phi_alpha = jnp.array(Phi_alpha)
    j_C = [jnp.array(C0), jnp.array(C1), jnp.array(C2), jnp.array(C3), jnp.array(C4)]
    j_Wxy1 = jnp.array(Wxy1)

    # ====================================================================
    # --- 3. CORE MATHEMATICS (JAX Computational Graph) ---
    # ====================================================================

    def compute_total_strain_energy(u_vec, alpha_vec):
        """
        Calculates the total strain energy of the shell element given
        nodal displacements (u_vec) and EAS internal parameters (alpha_vec).
        """
        # Interpolate physical fields based on nodal degrees of freedom
        U = jnp.einsum('n, ni... -> i...', u_vec, j_Phi)
        dU = jnp.einsum('n, nid... -> id...', u_vec, j_dPhi)

        u_phys, v_phys = U[0:3], U[3:6]
        du_phys, dv_phys = dU[0:3], dU[3:6]

        # Evaluate the EAS enhancement term (mu)
        mu = jnp.einsum('n, n... -> ...', alpha_vec, j_Phi_alpha[:, 0])

        # Compute standard strain vectors (constant, linear, and quadratic z-dependencies)
        eps0, eps1, eps2 = fsdt7_total_strain_vector_jax(shell.mid_surface_geometry,
                                                         u_phys, du_phys, v_phys, dv_phys,
                                                         xi1, xi2)

        # Inject the EAS field into the transverse normal strain component (epsilon_33 / index 2)
        eps1 = eps1.at[2].add(mu)

        # Compute stress resultants (L0, L1, L2) using pre-integrated stiffness matrices
        L0 = (jnp.einsum('ijxy, jxy->ixy', j_C[0], eps0) +
              jnp.einsum('ijxy, jxy->ixy', j_C[1], eps1) +
              jnp.einsum('ijxy, jxy->ixy', j_C[2], eps2))

        L1 = (jnp.einsum('ijxy, jxy->ixy', j_C[1], eps0) +
              jnp.einsum('ijxy, jxy->ixy', j_C[2], eps1) +
              jnp.einsum('ijxy, jxy->ixy', j_C[3], eps2))

        L2 = (jnp.einsum('ijxy, jxy->ixy', j_C[2], eps0) +
              jnp.einsum('ijxy, jxy->ixy', j_C[3], eps1) +
              jnp.einsum('ijxy, jxy->ixy', j_C[4], eps2))

        # Final numerical integration of the strain energy over the mid-surface
        energy = 0.5 * (jnp.einsum('ixy, ixy, xy->', L0, eps0, j_Wxy1) +
                        jnp.einsum('ixy, ixy, xy->', L1, eps1, j_Wxy1) +
                        jnp.einsum('ixy, ixy, xy->', L2, eps2, j_Wxy1))
        return energy

    # ====================================================================
    # OPTION 1: INTERNAL FORCE ONLY (For Residuals / Line Searches)
    # ====================================================================
    @jax.jit
    def get_internal_force(u_dof):
        """
        Computes ONLY the internal force vector via static condensation of the EAS parameters.
        Highly optimized by JAX XLA as it avoids computing the full K_uu tangent matrix.
        """
        alpha_zero = jnp.zeros(n_alpha)

        # 1. Compute EAS residual force (F_alpha) and EAS stiffness (K_aa) at alpha = 0
        F_alpha_raw = jax.grad(compute_total_strain_energy, argnums=1)(u_dof, alpha_zero)
        K_aa = jax.hessian(compute_total_strain_energy, argnums=1)(u_dof, alpha_zero)

        # 2. Solve for the internal equilibrium EAS parameters (alpha_eq)
        # alpha_eq = - inv(K_aa) * F_alpha_raw
        alpha_eq = jnp.linalg.solve(K_aa, -F_alpha_raw)

        # 3. Return the energy gradient with respect to u, evaluated at the correct alpha
        F_int_condensed = jax.grad(compute_total_strain_energy, argnums=0)(u_dof, alpha_eq)

        return F_int_condensed

    # ====================================================================
    # OPTION 2: FORCE AND TANGENT MATRIX (For Newton-Raphson Iterations)
    # ====================================================================
    @jax.jit
    def get_internal_force_and_tangent_matrix(u_dof):
        """
        Computes both the internal force vector and the statically condensed tangent stiffness matrix.
        Used when the solver needs to take a full Newton step.
        """
        alpha_zero = jnp.zeros(n_alpha)

        # Re-solve the internal EAS equilibrium state
        F_alpha_raw = jax.grad(compute_total_strain_energy, argnums=1)(u_dof, alpha_zero)
        K_aa = jax.hessian(compute_total_strain_energy, argnums=1)(u_dof, alpha_zero)
        alpha_eq = jnp.linalg.solve(K_aa, -F_alpha_raw)

        # Compute internal force (Gradient w.r.t U)
        F_int_condensed = jax.grad(compute_total_strain_energy, argnums=0)(u_dof, alpha_eq)

        # Extract the full block Hessian at the equilibrium state
        hessian_func = jax.hessian(compute_total_strain_energy, argnums=(0, 1))
        (K_uu, K_ua), (K_au, _) = hessian_func(u_dof, alpha_eq)

        # Perform Schur Complement condensation to obtain the effective tangent matrix
        # K_tan = K_uu - K_ua * inv(K_aa) * K_au
        K_tan_condensed = K_uu - jnp.dot(K_ua, jnp.linalg.solve(K_aa, K_au))

        return F_int_condensed, K_tan_condensed

    # Return the compiled JAX functions to be hooked into the non-linear solver
    return get_internal_force, get_internal_force_and_tangent_matrix