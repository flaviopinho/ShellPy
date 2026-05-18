"""
==============================================================================
ShellPy Test Suite: Formulation Verification
==============================================================================
This module validates the tensorial Koiter formulation for large rotations
against the classical numerical integration of the Sanders-Koiter shell theory,
and verifies the analytical tensor derivatives using Automatic Differentiation (JAX).

It ensures that the strain energy, internal force vectors (residuals), and
tangent stiffness matrices (Jacobians) calculated via exact tensor contraction
match the ones obtained through standard Gaussian numerical integration and JAX AD,
within an acceptable numerical tolerance limit.
==============================================================================
"""

import numpy as np
import sympy as sym
import jax

from shellpy.sanders_koiter import strain_energy_internal_force_and_tangent_matrix_jax

# Enforce double precision in JAX (Critical for Finite Element numerical stability)
jax.config.update("jax_enable_x64", True)

from shellpy.continuationpy.continuation import Continuation
from shellpy.sanders_koiter.internal_force_and_tangent_stiffness import tangent_stiffness_matrix, internal_force_vector
from shellpy.sanders_koiter.strain_energy import strain_energy
from shellpy.utils.residue_jacobian_stability import shell_stability, shell_jacobian, shell_residue
from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy.displacement_expansion import pinned

from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.koiter_tensor import koiter_load_energy, fast_koiter_strain_energy, koiter_strain_energy_large_rotations
from shellpy.tensor_derivatives import tensor_derivative

from shellpy.shell_loads.shell_conservative_load import PressureLoad
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_


def J_int_tensor(J_int, u, *args):
    """
    Contracts the stiffness tensor components with the displacement vector.
    """
    index_labels = "abcdefghijklmnopqrstuvwxyz"
    J_int_tot = J_int[0] + sum(
        np.einsum(
            f"ij{index_labels[:len(t.shape) - 2]},{','.join(index_labels[:len(t.shape) - 2])}->ij",
            t, *[u] * (len(t.shape) - 2), optimize=True
        )
        for t in J_int[1:]
    )
    return J_int_tot


def F_int_tensor(F_int, u, *args):
    """
    Contracts the internal force tensor components with the displacement vector.
    """
    index_labels = "abcdefghijklmnopqrstuvwxyz"
    F_int_tot = sum(
        np.einsum(
            f"i{index_labels[:len(t.shape) - 1]},{','.join(index_labels[:len(t.shape) - 1])}->i",
            t, *[u] * (len(t.shape) - 1), optimize=True
        )
        for t in F_int
    )
    return F_int_tot


def U_int_tensor(U_int, u, *args):
    """
    Full contraction of the strain energy tensor with the displacement vector.
    Leaves 0 free indices (Returns a scalar numeric value).
    """
    index_labels = "abcdefghijklmnopqrstuvwxyz"
    U_int_tot = sum(
        np.einsum(
            f"{index_labels[:len(t.shape)]},{','.join(index_labels[:len(t.shape)])}->",
            t, *[u] * len(t.shape), optimize=True
        )
        for t in U_int
    )
    return float(U_int_tot)


def _test_sanders_koiter_vs_tensorial_formulation():
    """
    Unit test comparing the numerical Sanders-Koiter formulation,
    the Tensorial Koiter approach, and JAX Automatic Differentiation.
    """
    # --- Geometric and mechanical parameters ---
    R = 0.1  # radius of the mid-surface (m)
    a = 0.1  # length in xi1-direction (m)
    b = 0.1  # length in xi2-direction (m)
    h = 0.0001  # shell thickness (m)
    E = 1  # Young’s modulus (Pa)
    nu = 0.3  # Poisson’s ratio
    density = 2  # Material density

    # --- Definition of rectangular mid-surface domain ---
    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    # --- Expansion size: number of modes for each displacement component ---
    expansion_size = {"u1": (3, 3), "u2": (3, 3), "u3": (3, 3)}

    # --- Boundary conditions: simply supported (S) on all sides ---
    boundary_conditions_u1 = {"xi1": ("S", "S"), "xi2": ("S", "S")}
    boundary_conditions_u2 = {"xi1": ("S", "S"), "xi2": ("S", "S")}
    boundary_conditions_u3 = {"xi1": ("S", "S"), "xi2": ("S", "S")}

    boundary_conditions = {
        "u1": boundary_conditions_u1,
        "u2": boundary_conditions_u2,
        "u3": boundary_conditions_u3,
    }

    # --- Creation of modal displacement field using eigenfunction expansion ---
    displacement_field = EigenFunctionExpansion(expansion_size, edges, boundary_conditions)

    # --- Symbolic definition of mid-surface geometry ---
    # R_ represents the position vector of a point on the mid-surface
    R_ = sym.Matrix([
        xi1_,
        xi2_,
        sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)
    ])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    load = PressureLoad(1)
    thickness = ConstantThickness(h)
    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, density)

    # Initialize the Shell object
    shell = Shell(mid_surface_geometry, thickness, edges, material, displacement_field, load)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Set random seed for reproducible test results
    np.random.seed(42)

    # Random zero-centered displacements.
    # Scaled down to prevent higher-order terms (U4) from causing numerical overflow.
    U_test = np.random.randn(n_dof) * 1e-1

    # 1. Strain Energy Tensor Coefficients (Analytical Koiter)
    U_ext = koiter_load_energy(shell)
    U2_int, U3_int, U4_int = koiter_strain_energy_large_rotations(shell)

    # 2. Strain Energy Tensor Coefficients (Numerical Sanders-Koiter)
    U2_sk, U3_sk, U4_sk = strain_energy(shell)

    div = E * h ** 2

    # First Derivatives (Internal Forces) using analytical tensors
    F_ext = tensor_derivative(U_ext, 0)
    F2_int = tensor_derivative(U2_int, 0)
    F3_int = tensor_derivative(U3_int, 0)
    F4_int = tensor_derivative(U4_int, 0)

    # Second Derivatives (Tangent Stiffness / Jacobians)
    J2_int = tensor_derivative(F2_int, 1)
    J3_int = tensor_derivative(F3_int, 1)
    J4_int = tensor_derivative(F4_int, 1)

    # =====================================================================
    # AVALIAÇÃO EM U_test
    # =====================================================================

    # A. Classical Numerical Formulation (Sanders-Koiter)
    F_sk = internal_force_vector(U_test, shell)
    J_sk = tangent_stiffness_matrix(U_test, shell)
    U_sk_eval = U_int_tensor((U2_sk, U3_sk, U4_sk), U_test)

    # B. Analytical Tensorial Formulation (Koiter)
    F_tensor = F_int_tensor((F2_int, F3_int, F4_int), U_test)
    J_tensor = J_int_tensor((J2_int, J3_int, J4_int), U_test)
    U_tensor = U_int_tensor((U2_int, U3_int, U4_int), U_test)

    # C. JAX Automatic Differentiation Formulation
    # Freeze the analytical tensors into JAX memory

    get_U_F_and_J_jax = strain_energy_internal_force_and_tangent_matrix_jax(shell)
    U_jax, F_jax, J_jax = get_U_F_and_J_jax(U_test)

    # =====================================================================
    # ERROR METRICS & ASSERTIONS
    # =====================================================================

    # 1. Classical vs Tensorial (Integration Error)
    erro_U_sk_tensor = abs(U_sk_eval - U_tensor)
    erro_F_sk_tensor = np.linalg.norm(F_sk - F_tensor)
    erro_J_sk_tensor = np.linalg.norm(J_sk - J_tensor)

    # 2. Tensorial vs JAX (Analytical Derivative Error - Should be ~0)
    erro_U_jax = abs(U_sk_eval - U_jax)
    erro_F_jax = np.linalg.norm(F_sk - F_jax)
    erro_J_jax = np.linalg.norm(J_sk - J_jax)

    print("=" * 75)
    print(" MODEL COMPARISON: Numerical vs Tensorial vs JAX AutoGrad")
    print("=" * 75)

    print(f"Strain Energy (U_sk):     {U_sk_eval:.6e}")
    print(f"Strain Energy (U_tensor): {U_tensor:.6e}")
    print(f"Strain Energy (U_jax):    {U_jax:.6e}")
    print(f"Error (SK vs Tensor):     {erro_U_sk_tensor:.6e}")
    print(f"Error (Tensor vs JAX):    {erro_U_jax:.6e} \n")

    print(f"Force Norm (F_sk):        {np.linalg.norm(F_sk):.6e}")
    print(f"Force Norm (F_tensor):    {np.linalg.norm(F_tensor):.6e}")
    print(f"Force Norm (F_jax):       {np.linalg.norm(F_jax):.6e}")
    print(f"Error (SK vs Tensor):     {erro_F_sk_tensor:.6e}")
    print(f"Error (Sk vs JAX):    {erro_F_jax:.6e} \n")

    print(f"Stiffness Norm (J_sk):    {np.linalg.norm(J_sk):.6e}")
    print(f"Stiffness Norm (J_tensor):{np.linalg.norm(J_tensor):.6e}")
    print(f"Stiffness Norm (J_jax):   {np.linalg.norm(J_jax):.6e}")
    print(f"Error (SK vs Tensor):     {erro_J_sk_tensor:.6e}")
    print(f"Error (Sk vs JAX):    {erro_J_jax:.6e} ")
    print("=" * 75)

    # Unit Test Assertions
    # 1. Integration vs Tensorial (Allows small numerical integration variance)
    np.testing.assert_allclose(U_sk_eval, U_tensor, rtol=1e-3, atol=1e-10,
                               err_msg="Strain energy mismatch between SK and Tensor.")
    np.testing.assert_allclose(F_sk, F_tensor, rtol=1e-2, atol=1e-6,
                               err_msg="Internal force mismatch between SK and Tensor.")
    np.testing.assert_allclose(J_sk, J_tensor, rtol=1e-2, atol=1e-2,
                               err_msg="Tangent stiffness mismatch between SK and Tensor.")

    # 2. SK vs JAX
    np.testing.assert_allclose(U_sk_eval, U_jax, rtol=1e-6, atol=1e-6,
                               err_msg="JAX energy mismatch. Contraction logic is flawed.")
    np.testing.assert_allclose(F_sk, F_jax, rtol=1e-6, atol=1e-6,
                               err_msg="JAX Force mismatch. Tensor derivatives are incorrect.")
    np.testing.assert_allclose(J_sk, J_jax, rtol=1e-2, atol=1e-2,
                               err_msg="JAX Jacobian mismatch. Tensor derivatives are incorrect.")

    print(">> TEST PASSED: Numerical, Tensorial, and JAX formulations are equivalent.")


if __name__ == "__main__":
    # Execute the test
    _test_sanders_koiter_vs_tensorial_formulation()