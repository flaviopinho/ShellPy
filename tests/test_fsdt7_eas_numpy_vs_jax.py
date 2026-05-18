import time
import numpy as np
import jax
import jax.numpy as jnp
import sympy as sym

from shellpy import RectangularMidSurfaceDomain, xi1_, xi2_, MidSurfaceGeometry, Shell, ConstantThickness
from shellpy.expansions import EigenFunctionExpansion
from shellpy.fsdt7_eas.EAS_expansion import EasExpansion
from shellpy.fsdt7_eas_jax.jax_strain_energy import fsdt7_strain_energy_internal_force_tangent_matrix_jax
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
from shellpy.materials import IsotropicHomogeneousLinearElasticMaterial

from shellpy.fsdt7_eas.internal_force_vector import internal_force_vector as internal_force_numpy
from shellpy.fsdt7_eas.tangent_stiffness_matrix import tangent_stiffness_matrix as tangent_stiffness_numpy

# Ensure double precision (Critical for stability in finite elements)
jax.config.update("jax_enable_x64", True)


def test_fsdt7_eas_numpy_vs_jax():
    """
    Test function to validate the JAX implementation of FSDT7 EAS
    against the original NumPy implementation.
    """
    print("\n=== STARTING VALIDATION TEST: FSDT7_EAS vs FSDT7_EAS_JAX ===")

    # =========================================================================
    # 1. GEOMETRY AND MESH INITIALIZATION
    # =========================================================================
    a, b, h = 1.0, 1.0, 0.1
    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    expansion_size = {"u1": (2, 2), "u2": (2, 1), "u3": (1, 2),
                      "v1": (2, 2), "v2": (1, 2), "v3": (2, 1)}
    boundary_conditions = {k: {"xi1": ("S", "S"), "xi2": ("S", "S")} for k in expansion_size}

    displacement_field = EigenFunctionExpansion(expansion_size, edges, boundary_conditions)
    eas_field = EasExpansion({"eas": (2, 2)}, edges, {"eas": {"xi1": ("F", "F"), "xi2": ("F", "F")}},
                             remove_constant_mode=True)

    R_ = sym.Matrix([
        xi1_, xi2_,
        (xi1_ - a / 2) ** 2 + (xi2_ - b / 2) ** 2 - (xi1_ - a / 2) * (xi2_ - b / 2),
    ])

    mid_surface_geometry = MidSurfaceGeometry(R_)
    material = IsotropicHomogeneousLinearElasticMaterial(E=10, nu=0.3, density=1550)

    shell = Shell(mid_surface_geometry, ConstantThickness(h), edges, material, displacement_field, None)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    nx, ny, nz = 10, 10, 5

    # Use a fixed seed to ensure reproducibility in tests
    np.random.seed(42)
    u_random = np.random.rand(n_dof)

    # =========================================================================
    # 2. EVALUATION WITH ORIGINAL LIBRARY (NUMPY)
    # =========================================================================
    print("\nComputing with NumPy (fsdt7_eas)...")
    t0 = time.perf_counter()
    F_int_np = internal_force_numpy(u_random, shell, eas_field, nx, ny, nz)
    K_tan_np = tangent_stiffness_numpy(u_random, shell, eas_field, nx, ny, nz)
    print(f"NumPy Time: {time.perf_counter() - t0:.4f} s")

    # =========================================================================
    # 3. JAX (Offline Phase / Factory Initialization)
    # =========================================================================
    print("\nPreparing JAX Cache via Factory...")
    t0 = time.perf_counter()
    _, internal_forces_and_tangent_matrix_jax = fsdt7_strain_energy_internal_force_tangent_matrix_jax(
        shell, eas_field, nx, ny, nz, gauss_weights_simple_integral
    )
    print(f"Factory Time: {time.perf_counter() - t0:.4f} s")

    # =========================================================================
    # 4. EVALUATION WITH NEW LIBRARY (JAX)
    # =========================================================================
    print("\nComputing and compiling with JAX (fsdt7_eas_jax)...")

    # First call: Triggers the JIT Compiler
    t0 = time.perf_counter()
    F_int_jax, K_tan_jax = internal_forces_and_tangent_matrix_jax(jnp.array(u_random))

    # Synchronizes and pulls data from GPU/CPU to standard NumPy arrays
    F_int_jax = np.array(F_int_jax)
    K_tan_jax = np.array(K_tan_jax)
    print(f"JAX Time (Compilation + 1st Execution): {time.perf_counter() - t0:.4f} s")

    # Second call: Measure actual execution time in Hot-Loop
    t0 = time.perf_counter()
    F_int_hot, K_tan_hot = internal_forces_and_tangent_matrix_jax(jnp.array(u_random))
    F_int_hot.block_until_ready()  # Wait for async execution to finish to measure correctly
    print(f"JAX Time (Cached Hot-Loop Execution): {time.perf_counter() - t0:.6f} s")

    # =========================================================================
    # 5. MATHEMATICAL RESULTS COMPARISON (ASSERTIONS)
    # =========================================================================
    print("\n--- COMPARISON RESULTS ---")

    error_f = np.max(np.abs(F_int_np - F_int_jax))
    error_k = np.max(np.abs(K_tan_np - K_tan_jax))

    print(f"Max Error in Internal Forces: {error_f:.2e}")
    print(f"Max Error in Tangent Stiffness: {error_k:.2e}")

    # Explicit Assertions to make this a standard test
    # If the error is higher than the tolerance, the test will throw an AssertionError and fail
    np.testing.assert_allclose(
        F_int_jax, F_int_np,
        atol=1e-6, rtol=1e-6,
        err_msg="Internal forces mismatch between NumPy and JAX!"
    )

    np.testing.assert_allclose(
        K_tan_jax, K_tan_np,
        atol=1e-6, rtol=1e-6,
        err_msg="Tangent stiffness mismatch between NumPy and JAX!"
    )

    print("\n[ SUCCESS ] - JAX implementation matches NumPy implementation perfectly!")


# This allows you to run the test script directly from the terminal
# using `python test_file.py` in addition to running it via PyTest.
if __name__ == "__main__":
    test_fsdt7_eas_numpy_vs_jax()