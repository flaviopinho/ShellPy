import jax
import jax.numpy as jnp
import numpy as np

from shellpy import Shell
from shellpy.numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_y, \
    n_integral_default_z
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
from shellpy.numeric_integration.integral_weights import double_integral_weights
from shellpy.sanders_koiter.jax_total_strain_vector import total_strain_vector_jax
from shellpy.sanders_koiter.plane_stress_constitutive_matrix_in_material_frame import \
    constitutive_matrix_in_material_frame
from shellpy.sanders_koiter.plane_stress_constitutive_matrix_in_shell_frame import \
    plane_stress_constitutive_matrix_in_shell_frame

# Enforce double precision in JAX (Critical for Finite Element / Mechanics numerical stability)
jax.config.update("jax_enable_x64", True)


def strain_energy_internal_force_and_tangent_matrix_jax(shell: Shell,
                                                        n_x=n_integral_default_x,
                                                        n_y=n_integral_default_y,
                                                        n_z=n_integral_default_z,
                                                        integral_method=gauss_weights_simple_integral):
    """
    JAX-based Matrix-Free factory for the Sanders-Koiter shell formulation.

    This function pre-computes static geometry and basis functions (Offline Phase)
    and returns a highly optimized, JIT-compiled JAX function.
    The compiled function takes the displacement vector (u) and returns the
    Strain Energy, Internal Force vector (Gradient), and Tangent Stiffness matrix (Hessian).
    """

    # =========================================================================
    # 1. PRE-PROCESSING (OFFLINE PHASE) - O(N) Memory Footprint
    # =========================================================================
    print("Pre-computing geometry and Interpolation Tensors...")

    # Get integration points and weights for the mid-surface area integral
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)

    # Determine thickness and through-the-thickness integration points
    h = shell.thickness(xi1, xi2)
    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    n_xy = np.shape(xi1)
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # 1.1 Interpolation Matrices (Ritz Basis Functions)
    # Instead of crossing DOFs, we only evaluate shape functions at Gauss points.
    # Format: (n_dof, component_i, Nx, Ny)
    Phi = np.zeros((n_dof, 3) + n_xy)
    dPhi = np.zeros((n_dof, 3, 2) + n_xy)  # First derivatives w.r.t xi1, xi2
    ddPhi = np.zeros((n_dof, 3, 2, 2) + n_xy)  # Second derivatives w.r.t xi1, xi2

    for n in range(n_dof):
        # Displacements u1, u2, u3
        Phi[n] = shell.displacement_expansion.shape_function(n, xi1, xi2, 0, 0)
        # First Covariant/Partial Derivatives
        dPhi_n = shell.displacement_expansion.shape_function_first_derivatives(n, xi1, xi2)
        dPhi[n] = dPhi_n
        # Second Covariant/Partial Derivatives
        ddPhi_n = shell.displacement_expansion.shape_function_second_derivatives(n, xi1, xi2)
        ddPhi[n] = ddPhi_n

    # 1.2 Geometry and Constitutive Tensors
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    Wxy1 = sqrtG * Wxy
    det_shifter = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    # Compute constitutive matrix in material frame and transform to shell frame
    C_material = constitutive_matrix_in_material_frame(shell.mid_surface_geometry, shell.material, (xi1, xi2, xi3))
    if C_material.ndim == 2:
        C_material = np.einsum('ij, xyz->ijxyz', C_material, xi3 ** 0)
    C_shell = plane_stress_constitutive_matrix_in_shell_frame(shell.mid_surface_geometry, C_material, (xi1, xi2, xi3))

    # Through-the-thickness integration to obtain Membrane (C0), Coupling (C1), and Bending (C2) stiffnesses
    C0 = np.einsum('ijxyz, xyz->ijxy', C_shell, xi3 ** 0 * det_shifter * Wz)
    C1 = np.einsum('ijxyz, xyz->ijxy', C_shell, xi3 ** 1 * det_shifter * Wz)
    C2 = np.einsum('ijxyz, xyz->ijxy', C_shell, xi3 ** 2 * det_shifter * Wz)

    # =========================================================================
    # 2. JAX CONVERSION
    # =========================================================================
    # Freeze the pre-computed NumPy arrays into JAX DeviceArrays.
    # These will be baked into the XLA compiled closure.
    j_Phi = jnp.array(Phi)
    j_dPhi = jnp.array(dPhi)
    j_ddPhi = jnp.array(ddPhi)
    j_C0 = jnp.array(C0)
    j_C1 = jnp.array(C1)
    j_C2 = jnp.array(C2)
    j_Wxy1 = jnp.array(Wxy1)

    # =========================================================================
    # 3. ENERGY FUNCTION (ONLINE PHASE) - AutoGrad over Physical Fields
    # =========================================================================
    def compute_total_strain_energy(u_vec):
        """Pure mathematical functional mapping modal amplitudes to total strain energy."""

        # 3.1 Physical Field Reconstruction (Interpolation)
        # Maps the 1D generalized coordinate vector (u_vec) to the 2D physical grid
        u_phys = jnp.einsum('n, ni... -> i...', u_vec, j_Phi)
        du_phys = jnp.einsum('n, nia... -> ia...', u_vec, j_dPhi)
        ddu_phys = jnp.einsum('n, niab... -> iab...', u_vec, j_ddPhi)

        # 3.2 Kinematics (Sanders-Koiter)
        # Calculates membrane (eps0) and bending (eps1) strains including large rotations
        eps0, eps1 = total_strain_vector_jax(shell.mid_surface_geometry, u_phys, du_phys, ddu_phys, xi1, xi2)

        # 3.3 Stress Resultants
        # N = Membrane Forces, M = Bending Moments
        N = jnp.einsum('ijxy, jxy->ixy', j_C0, eps0) + jnp.einsum('ijxy, jxy->ixy', j_C1, eps1)
        M = jnp.einsum('ijxy, jxy->ixy', j_C1, eps0) + jnp.einsum('ijxy, jxy->ixy', j_C2, eps1)

        # 3.4 Total Strain Energy Density Integration
        # U = 1/2 * Integral(N*eps0 + M*eps1) dA
        energy_density = 0.5 * (jnp.einsum('ixy, ixy -> xy', N, eps0) + jnp.einsum('ixy, ixy -> xy', M, eps1))
        return jnp.sum(energy_density * j_Wxy1)

    # =========================================================================
    # 4. UNIFIED COMPILATION (Energy, Gradient, and Hessian)
    # =========================================================================
    print("Compiling JAX functions")

    # @jax.jit compiles this function into highly optimized C++/GPU instructions via XLA.
    # By calling the function, its grad, and its hessian together, the XLA compiler
    # uses Common Subexpression Elimination (CSE) to share the kinematics calculations,
    # drastically reducing execution time during Newton-Raphson iterations.
    @jax.jit
    def func_grad_and_hessian(u_vec):
        return (
            compute_total_strain_energy(u_vec),
            jax.grad(compute_total_strain_energy)(u_vec),
            jax.hessian(compute_total_strain_energy)(u_vec)
        )

    return func_grad_and_hessian