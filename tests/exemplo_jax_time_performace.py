"""
==============================================================================
ShellPy Benchmark Suite: Performance Comparison (Fair & Rigorous)
==============================================================================
This module measures the computational performance of three non-linear
shell formulations:
1. Classical Numerical (Sanders-Koiter) - Matricial
2. Tensorial Koiter (Exact contractions via numpy.einsum)
3. JAX AutoGrad (Full Numerical Integration Model compiled with XLA)

It strictly separates the Offline phase (initialization, integration, and
JIT compilation) from the Online phase (Newton-Raphson evaluation loop).
==============================================================================
"""

import time
import numpy as np
import sympy as sym
import jax
import jax.numpy as jnp

from shellpy.sanders_koiter import strain_energy_internal_force_and_tangent_matrix_jax

# Enforce double precision in JAX (Critical for MEF)
jax.config.update("jax_enable_x64", True)

# --- General ShellPy Imports ---
from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.koiter_tensor import koiter_strain_energy_large_rotations
from shellpy.tensor_derivatives import tensor_derivative
from shellpy.shell_loads.shell_conservative_load import PressureLoad
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
from shellpy.numeric_integration.integral_weights import double_integral_weights

# --- Imports from the refactored SK formulation (Matricial) ---
from shellpy.sanders_koiter._compute_constant_shell_matrices import compute_constant_shell_matrices
from shellpy.sanders_koiter._compute_displacement_dependent_matrices import compute_displacement_dependent_matrices


# =============================================================================
# CONTRACTION HELPER FUNCTIONS (For Tensorial Koiter)
# =============================================================================
def J_int_tensor(J_int, u):
    index_labels = "abcdefghijklmnopqrstuvwxyz"
    J_int_tot = J_int[0] + sum(
        np.einsum(f"ij{index_labels[:len(t.shape) - 2]},{','.join(index_labels[:len(t.shape) - 2])}->ij",
                  t, *[u] * (len(t.shape) - 2), optimize=True) for t in J_int[1:]
    )
    return J_int_tot


def F_int_tensor(F_int, u):
    index_labels = "abcdefghijklmnopqrstuvwxyz"
    F_int_tot = sum(
        np.einsum(f"i{index_labels[:len(t.shape) - 1]},{','.join(index_labels[:len(t.shape) - 1])}->i",
                  t, *[u] * (len(t.shape) - 1), optimize=True) for t in F_int
    )
    return F_int_tot


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================
def run_performance_benchmark():
    print("=" * 70)
    print(" SHELLPY PERFORMANCE BENCHMARK")
    print("=" * 70)

    # --- 1. SETTING UP THE SHELL DOMAIN ---
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.0001
    E = 1
    nu = 0.3
    density = 2

    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    # --- PARÂMETRO CRÍTICO PARA ESCALABILIDADE (Tamanho da Expansão) ---
    modes = 6
    expansion_size = {"u1": (modes, modes), "u2": (modes, modes), "u3": (modes, modes)}

    bc = {"xi1": ("S", "S"), "xi2": ("S", "S")}
    boundary_conditions = {"u1": bc, "u2": bc, "u3": bc}

    displacement_field = EigenFunctionExpansion(expansion_size, edges, boundary_conditions)
    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    load = PressureLoad(1)
    thickness = ConstantThickness(h)
    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, density)

    shell = Shell(mid_surface_geometry, thickness, edges, material, displacement_field, load)
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    print(f">> Initializing Shell Model with {n_dof} Degrees of Freedom (DOF)\n")

    # =========================================================================
    # OFFLINE PHASE (Initialization & Integration)
    # =========================================================================
    print("--- Running Offline Phase (Pre-computation) ---")

    # 1. Sanders-Koiter matrix formulation
    t0 = time.perf_counter()
    Wxy1, C0, C1, C2, eps0_lin, eps1_lin, eps0_nl, eps1_nl = compute_constant_shell_matrices(
        shell, 10, 10, 5, gauss_weights_simple_integral
    )
    t_off_sk = time.perf_counter() - t0
    print(f"Sanders-Koiter matrix formulation prepared in     {t_off_sk:.4f} s")

    # 2. Tensorial Koiter
    t0 = time.perf_counter()
    U2_int, U3_int, U4_int = koiter_strain_energy_large_rotations(shell, 10, 10, 5, gauss_weights_simple_integral)
    div = E * h ** 2

    F2_int = tensor_derivative(U2_int, 0) * h / div
    F3_int = tensor_derivative(U3_int, 0) * h ** 2 / div
    F4_int = tensor_derivative(U4_int, 0) * h ** 3 / div

    J2_int = tensor_derivative(F2_int, 1)
    J3_int = tensor_derivative(F3_int, 1)
    J4_int = tensor_derivative(F4_int, 1)
    t_off_tensor = time.perf_counter() - t0
    print(f"Tensorial Koiter formulation prepared in        {t_off_tensor:.4f} s")

    # 3. JAX AutoGrad (USING FULL NUMERICAL MODEL)
    t0 = time.perf_counter()
    get_F_and_J_jax = strain_energy_internal_force_and_tangent_matrix_jax(
        shell, 10, 10, 5, gauss_weights_simple_integral
    )

    dummy_u = jnp.zeros(n_dof)
    U_jax, F_jax, K_jax = get_F_and_J_jax(dummy_u)
    t_off_jax = time.perf_counter() - t0
    print(f"JAX compiled in              {t_off_jax:.4f} s\n")

    # =========================================================================
    # ONLINE PHASE (Newton-Raphson Loop Simulation)
    # =========================================================================
    N_ITERATIONS = 50
    print(f"--- Running Online Phase (Averaging {N_ITERATIONS} Newton-Raphson Iterations) ---")

    np.random.seed(42)
    # Cria uma simulação de "passos" para desarmar o cache matricial
    base_u = np.random.randn(n_dof) * 1e-4
    u_steps = [base_u + (np.random.randn(n_dof) * 1e-6) for _ in range(N_ITERATIONS)]

    # 1. Classical SK (Matricial)
    t0 = time.perf_counter()
    for u in u_steps:
        _F_sk, _J_sk = compute_displacement_dependent_matrices(
            u, Wxy1, C0, C1, C2, eps0_lin, eps1_lin, eps0_nl, eps1_nl
        )
    t_on_sk = (time.perf_counter() - t0) / N_ITERATIONS

    # 2. Tensorial Koiter
    t0 = time.perf_counter()
    for u in u_steps:
        _F_tensor = F_int_tensor((F2_int, F3_int, F4_int), u)
        _J_tensor = J_int_tensor((J2_int, J3_int, J4_int), u)
    t_on_tensor = (time.perf_counter() - t0) / N_ITERATIONS

    # 3. JAX AutoGrad (Full Physics Model)
    t0 = time.perf_counter()
    for u in u_steps:
        j_u = jnp.array(u)
        _, _F_jax, _J_jax = get_F_and_J_jax(j_u)
    t_on_jax = (time.perf_counter() - t0) / N_ITERATIONS

    # =========================================================================
    # PERFORMANCE RESULTS TABLE
    # =========================================================================
    print("=" * 70)
    print(f"{'FORMULATION':<25} | {'OFFLINE TIME (s)':<18} | {'ONLINE TIME (s/iter)':<20}")
    print("-" * 70)
    print(f"{'SK Matrix formulation (Numpy)':<25} | {t_off_sk:<18.4f} | {t_on_sk:<20.6e}")
    print(f"{'Koiter Tensorial (Einsum)':<25} | {t_off_tensor:<18.4f} | {t_on_tensor:<20.6e}")
    print(f"{'JAX AutoGrad (Full Model)':<25} | {t_off_jax:<18.4f} | {t_on_jax:<20.6e}")
    print("=" * 70)

    # Calculate Speedups relative to the classical numerical approach
    speedup_tensor = t_on_sk / t_on_tensor
    speedup_jax = t_on_sk / t_on_jax

    print(f"\n[ONLINE SPEEDUP ANALYSIS]")
    print(f"- Tensorial Koiter is {speedup_tensor:.1f}x FASTER than SK Matricial.")
    print(f"- JAX AutoGrad is {speedup_jax:.1f}x FASTER than SK Matricial.")


if __name__ == "__main__":
    run_performance_benchmark()
