import time
import numpy as np
import jax
import sympy as sym

from shellpy import RectangularMidSurfaceDomain, xi1_, xi2_, MidSurfaceGeometry, Shell, ConstantThickness
from shellpy.expansions import EigenFunctionExpansion
from shellpy.fsdt6.strain_vector import linear_strain_vector, nonlinear_strain_vector
from shellpy.fsdt7_eas.EAS_expansion import EasExpansion
from shellpy.fsdt7_eas_jax import fsdt7_total_strain_vector_jax
from shellpy.materials import IsotropicHomogeneousLinearElasticMaterial

# Ensure double precision in JAX for exact validation against NumPy
jax.config.update("jax_enable_x64", True)


def test_fsdt7_strain_vector_numpy_vs_jax():
    """
    Test function to validate the JAX implementation of the FSDT7 total strain vector
    against the original NumPy implementation (combining linear and non-linear parts).
    """
    print("\n=== STARTING VALIDATION TEST: STRAIN VECTOR NumPy vs JAX ===")

    # =========================================================================
    # 1. GEOMETRY AND MESH INITIALIZATION
    # =========================================================================
    a, b, h = 1.0, 1.0, 0.1
    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    expansion_size = {"u1": (1, 1), "u2": (1, 1), "u3": (1, 1),
                      "v1": (1, 1), "v2": (1, 1), "v3": (1, 1)}

    boundary_conditions = {k: {"xi1": ("S", "S"), "xi2": ("S", "S")} for k in expansion_size}

    displacement_field = EigenFunctionExpansion(expansion_size, edges, boundary_conditions)
    eas_field = EasExpansion({"eas": (1, 1)}, edges, {"eas": {"xi1": ("F", "F"), "xi2": ("F", "F")}})

    R_ = sym.Matrix([
        xi1_, xi2_,
        (xi1_ - a / 2) ** 2 + (xi2_ - b / 2) ** 2 - (xi1_ - a / 2) * (xi2_ - b / 2),
    ])

    mid_surface_geometry = MidSurfaceGeometry(R_)
    material = IsotropicHomogeneousLinearElasticMaterial(E=10, nu=0.3, density=1550)

    shell = Shell(mid_surface_geometry, ConstantThickness(h), edges, material, displacement_field, None)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    n_alpha = eas_field.number_of_degrees_of_freedom()

    # Generate random displacement vector for testing (scaled down to represent small physical displacements)
    np.random.seed(42)
    u_random = np.random.rand(n_dof) * 1

    # Define a unique evaluation point to ensure exact consistency between both evaluations
    xi1_eval, xi2_eval = 0.5, 0.5

    # =========================================================================
    # 2. CLASSIC FORMULATION (NumPy)
    # =========================================================================
    print("\n--- Computing NumPy Strains ---")
    t0 = time.perf_counter()

    # Initialize linear strain arrays
    epsilon0_lin = np.zeros((n_dof, 6))
    epsilon1_lin = np.zeros((n_dof, 6))
    epsilon2_lin = np.zeros((n_dof, 6))

    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i], epsilon2_lin[i] = linear_strain_vector(
            shell.mid_surface_geometry, shell.displacement_expansion, i, xi1_eval, xi2_eval)

    # Initialize non-linear strain arrays
    epsilon0_nl = np.zeros((n_dof, n_dof, 6))
    epsilon1_nl = np.zeros((n_dof, n_dof, 6))
    epsilon2_nl = np.zeros((n_dof, n_dof, 6))

    for i in range(n_dof):
        for j in range(i, n_dof):
            res0, res1, res2 = nonlinear_strain_vector(
                shell.mid_surface_geometry, shell.displacement_expansion, i, j, xi1_eval, xi2_eval)

            # Symmetric population of the non-linear tensors
            epsilon0_nl[i, j], epsilon1_nl[i, j], epsilon2_nl[i, j] = res0, res1, res2
            epsilon0_nl[j, i], epsilon1_nl[j, i], epsilon2_nl[j, i] = res0, res1, res2

    # Calculate total strains (Linear + Non-Linear contributions)
    eps0_np = np.einsum('ia..., i->a...', epsilon0_lin, u_random) + \
              np.einsum('ija..., i, j->a...', epsilon0_nl, u_random, u_random)

    eps1_np = np.einsum('ia..., i->a...', epsilon1_lin, u_random) + \
              np.einsum('ija..., i, j->a...', epsilon1_nl, u_random, u_random)

    eps2_np = np.einsum('ia..., i->a...', epsilon2_lin, u_random) + \
              np.einsum('ija..., i, j->a...', epsilon2_nl, u_random, u_random)

    print(f"NumPy Time: {time.perf_counter() - t0:.4f} s")

    # =========================================================================
    # 3. JAX FORMULATION
    # =========================================================================
    print("--- Computing JAX Strains ---")
    t0 = time.perf_counter()

    Phi = np.zeros((n_dof, 6))
    dPhi = np.zeros((n_dof, 6, 2))

    # Evaluate shape functions and derivatives at the integration point
    for n in range(n_dof):
        Phi[n] = shell.displacement_expansion.shape_function(n, xi1_eval, xi2_eval)
        dPhi[n] = shell.displacement_expansion.shape_function_first_derivatives(n, xi1_eval, xi2_eval)

    # Reconstruct physical displacement and its derivative
    U = np.einsum('n, ni... -> i...', u_random, Phi)
    dU = np.einsum('n, nid... -> id...', u_random, dPhi)

    u_phys, v_phys = U[0:3], U[3:6]
    du_phys, dv_phys = dU[0:3], dU[3:6]

    # Compute total strains using the JAX optimized function
    eps0_jax, eps1_jax, eps2_jax = fsdt7_total_strain_vector_jax(
        shell.mid_surface_geometry, u_phys, du_phys, v_phys, dv_phys, xi1_eval, xi2_eval)

    print(f"JAX Time: {time.perf_counter() - t0:.4f} s")

    # =========================================================================
    # 4. RESULTS COMPARISON (ASSERTIONS)
    # =========================================================================
    print("\n--- STRAIN COMPARISON RESULTS ---")

    diff_eps0 = np.max(np.abs(eps0_np - np.array(eps0_jax)))
    diff_eps1 = np.max(np.abs(eps1_np - np.array(eps1_jax)))
    diff_eps2 = np.max(np.abs(eps2_np - np.array(eps2_jax)))

    print(f"Max difference in eps0 (Membrane/Shear): {diff_eps0:.5e}")
    print(f"Max difference in eps1 (Bending/Linear z): {diff_eps1:.5e}")
    print(f"Max difference in eps2 (Quadratic z):      {diff_eps2:.5e}")

    # Standard testing assertions
    # If the max difference is roughly half or double, it usually means the 1/2 factor inside
    # the JAX formulation or the symmetric double-summation in NumPy is misaligned.
    error_msg_hint = (
        "\n[DEBUG HINT] Check symmetry scaling: "
        "Since `epsilon_nl[i, j]` and `epsilon_nl[j, i]` are populated and summed over `i, j`, "
        "you might be double counting the non-linear part in NumPy. "
        "Verify if `nonlinear_strain_vector` already contains the 1/2 factor."
    )

    np.testing.assert_allclose(
        eps0_jax, eps0_np,
        atol=1e-10, rtol=1e-10,
        err_msg=f"eps0 mismatch between NumPy and JAX! {error_msg_hint}"
    )

    np.testing.assert_allclose(
        eps1_jax, eps1_np,
        atol=1e-10, rtol=1e-10,
        err_msg=f"eps1 mismatch between NumPy and JAX! {error_msg_hint}"
    )

    np.testing.assert_allclose(
        eps2_jax, eps2_np,
        atol=1e-10, rtol=1e-10,
        err_msg=f"eps2 mismatch between NumPy and JAX! {error_msg_hint}"
    )

    print("\n[ SUCCESS ] - All strain vectors match perfectly!")


if __name__ == "__main__":
    test_fsdt7_strain_vector_numpy_vs_jax()