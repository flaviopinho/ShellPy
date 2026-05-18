import time
import tracemalloc
import numpy as np
import jax
import jax.numpy as jnp
import sympy as sym

from shellpy import RectangularMidSurfaceDomain, xi1_, xi2_, MidSurfaceGeometry, Shell, ConstantThickness
from shellpy.cache_decorator import clear_cache
from shellpy.expansions import EigenFunctionExpansion
from shellpy.fsdt7_eas.EAS_expansion import EasExpansion
from shellpy.fsdt7_eas_jax.jax_strain_energy import fsdt7_strain_energy_internal_force_tangent_matrix_jax
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
from shellpy.materials import IsotropicHomogeneousLinearElasticMaterial

# --- Old library imports (NumPy) ---
from shellpy.fsdt7_eas.internal_force_vector import internal_force_vector as internal_force_numpy
from shellpy.fsdt7_eas.tangent_stiffness_matrix import tangent_stiffness_matrix as tangent_stiffness_numpy

# Ensure double precision in JAX for exact validation
jax.config.update("jax_enable_x64", True)


# =========================================================================
# MEMORY TRACKER UTILITY
# =========================================================================
class MemoryTracker:
    """
    Context manager to track peak memory allocation (in MB) of a specific block of code
    using Python's built-in tracemalloc library.
    """

    def __enter__(self):
        tracemalloc.start()
        tracemalloc.reset_peak()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mb = peak / (1024 * 1024)  # Convert bytes to Megabytes


def setup_problem(order_u, order_eas):
    """Creates the geometry, mesh, and fields according to the desired expansion order."""
    a, b, h = 1.0, 1.0, 0.1
    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    # Increasing the order increases the number of degrees of freedom
    expansion_size = {"u1": (order_u, order_u), "u2": (order_u, order_u), "u3": (order_u, order_u),
                      "v1": (order_u, order_u), "v2": (order_u, order_u), "v3": (order_u, order_u)}
    boundary_conditions = {k: {"xi1": ("S", "S"), "xi2": ("S", "S")} for k in expansion_size}

    displacement_field = EigenFunctionExpansion(expansion_size, edges, boundary_conditions)
    eas_field = EasExpansion({"eas": (order_eas, order_eas)}, edges, {"eas": {"xi1": ("F", "F"), "xi2": ("F", "F")}})

    R_ = sym.Matrix([
        xi1_, xi2_,
        (xi1_ - a / 2) ** 2 + (xi2_ - b / 2) ** 2 - (xi1_ - a / 2) * (xi2_ - b / 2),
    ])

    mid_surface_geometry = MidSurfaceGeometry(R_)
    material = IsotropicHomogeneousLinearElasticMaterial(E=10, nu=0.3, density=1550)

    shell = Shell(mid_surface_geometry, ConstantThickness(h), edges, material, displacement_field, None)

    # Integration points
    nx, ny, nz = 10, 10, 5

    return shell, eas_field, nx, ny, nz


def test_fsdt7_eas_benchmark_and_accuracy():
    """
    Test function that validates both the accuracy and benchmarks the performance
    (Time and Memory) of the JAX JIT implementation against the classic NumPy implementation.
    """
    print("\n============================================================")
    print("   PERFORMANCE, MEMORY & ACCURACY TEST: NUMPY vs JAX JIT      ")
    print("============================================================\n")

    # Expansion orders to test. Ex: 3x3, 4x4
    test_orders = [3]

    # Number of "Hot-Loop" iterations simulating Newton-Raphson steps
    n_iterations = 10

    shell = None
    for order in test_orders:
        clear_cache(shell)
        shell, eas_field, nx, ny, nz = setup_problem(order_u=order, order_eas=order)
        n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

        print(f"--- TESTING ORDER {order}x{order} | Degrees of Freedom (DOFs): {n_dof} ---")

        np.random.seed(42)
        u_vectors = [np.random.rand(n_dof) * 1e-2 for _ in range(n_iterations)]

        # ==========================================
        # NUMPY PHASE
        # ==========================================
        print("  [NumPy] Evaluating...")
        start_np = time.perf_counter()

        with MemoryTracker() as mem_np:
            F_np = None
            K_np = None
            for u in u_vectors:
                F_np = internal_force_numpy(u, shell, eas_field, nx, ny, nz)
                K_np = tangent_stiffness_numpy(u, shell, eas_field, nx, ny, nz)

        time_np = (time.perf_counter() - start_np) / n_iterations

        # ==========================================
        # JAX PHASE
        # ==========================================
        print("  [JAX] Factory (Pre-computing constant tensors)...")
        start_factory = time.perf_counter()

        with MemoryTracker() as mem_factory:
            get_f_jax, get_f_and_k_jax = fsdt7_strain_energy_internal_force_tangent_matrix_jax(
                shell, eas_field, nx, ny, nz, gauss_weights_simple_integral
            )

        time_factory = time.perf_counter() - start_factory

        print("  [JAX] Compiling XLA graph (1st Execution)...")
        start_compile = time.perf_counter()

        with MemoryTracker() as mem_compile:
            F_jax_first, K_jax_first = get_f_and_k_jax(jnp.array(u_vectors[0]))
            F_jax_first.block_until_ready()  # Force execution to finish

        time_compile = time.perf_counter() - start_compile

        print("  [JAX] Executing Hot-Loop...")
        start_jax_hot = time.perf_counter()

        with MemoryTracker() as mem_hot:
            F_jax = None
            K_jax = None
            for u in u_vectors:
                F_jax, K_jax = get_f_and_k_jax(jnp.array(u))
                F_jax.block_until_ready()

        time_jax_hot = (time.perf_counter() - start_jax_hot) / n_iterations

        # ==========================================
        # MATHEMATICAL VALIDATION (Assertions)
        # ==========================================
        F_jax_np = np.array(F_jax)
        K_jax_np = np.array(K_jax)

        erro_f = np.max(np.abs(F_np - F_jax_np))
        erro_k = np.max(np.abs(K_np - K_jax_np))

        np.testing.assert_allclose(
            F_jax_np, F_np,
            atol=1e-12, rtol=1e-6,
            err_msg=f"Internal forces mismatch at order {order}!"
        )

        np.testing.assert_allclose(
            K_jax_np, K_np,
            atol=1e-3, rtol=1e-3,
            err_msg=f"Tangent stiffness mismatch at order {order}!"
        )

        # ==========================================
        # RESULTS SUMMARY
        # ==========================================
        speedup = time_np / time_jax_hot if time_jax_hot > 0 else 0

        print(f"\n  Summary (DOFs={n_dof}):")
        print(f"  - Validation     : [ PASSED ] (Max Error F: {erro_f:.1e} | Max Error K: {erro_k:.1e})")
        print(f"  - NumPy          : {time_np:.5f} s/it  | Peak RAM: {mem_np.peak_mb:.2f} MB")
        print(f"  - JAX Hot-Loop   : {time_jax_hot:.5f} s/it  | Peak RAM: {mem_hot.peak_mb:.2f} MB")
        print(
            f"  - JAX Setup Cost : Factory {time_factory:.2f}s ({mem_factory.peak_mb:.2f} MB) | Compile {time_compile:.2f}s ({mem_compile.peak_mb:.2f} MB)")
        print(f"  - ACCELERATION   : JAX Hot-Loop is {speedup:.2f}x faster!\n")


if __name__ == "__main__":
    test_fsdt7_eas_benchmark_and_accuracy()