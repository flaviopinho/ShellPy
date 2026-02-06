"""
Free vibration analysis of functionally graded shallow shells
based on:

H. Matsunaga, "Free vibration and stability of functionally graded shallow
shells according to a 2D higher-order deformation theory",
Composite Structures.

This script reproduces the linear free-vibration results reported in
Table 2 of the reference paper and is structured to automatically run
multiple geometric curvature cases (flat, cylindrical, doubly curved)
using a unified solver pipeline.

Main steps:
1. Definition of geometry and curvature cases
2. Functionally graded material modeling
3. Kinematic expansion (FSDT5)
4. Energy-based formulation (kinetic and strain energies)
5. Assembly of mass and stiffness matrices
6. Solution of the generalized eigenvalue problem
7. Extraction of normalized and dimensional natural frequencies
"""

# ======================================================================
# Imports
# ======================================================================

import numpy as np
import sympy as sym
from scipy.linalg import eig

# ShellPy core
from shellpy import (
    Shell,
    ConstantThickness,
    RectangularMidSurfaceDomain,
    MidSurfaceGeometry,
    xi1_, xi2_,
    simply_supported_fsdt5,
)
from shellpy.cache_decorator import clear_cache

# Kinematic expansions
from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion

# Energy formulations (FSDT5)
from shellpy.fsdt5.kinetic_energy import kinetic_energy
from shellpy.fsdt5.strain_energy import quadratic_strain_energy

# Material model
from shellpy.materials.functionally_graded_material import FunctionallyGradedMaterial

# Automatic differentiation utilities
from shellpy.tensor_derivatives import tensor_derivative

# Plot
from exemples.paper_results.shell_mode import shell_mode


# ======================================================================
# Geometry factory
# ======================================================================

def midsurface_geometry(case, a, b, Rx=None, Ry=None):
    """
    Returns the symbolic mid-surface parametrization R(xi1, xi2).
    """
    if case == "plate":
        return sym.Matrix([xi1_, xi2_, 0])

    elif case == "cylindrical_x":
        return sym.Matrix([
            xi1_,
            xi2_,
            (xi1_ - a / 2) ** 2 / (2 * Rx),
        ])

    elif case == "doubly_curved":
        return sym.Matrix([
            xi1_,
            xi2_,
            (xi1_ - a / 2) ** 2 / (2 * Rx)
            + (xi2_ - b / 2) ** 2 / (2 * Ry),
        ])

    else:
        raise ValueError(f"Unknown geometry case: {case}")


# ======================================================================
# Main execution
# ======================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Numerical integration parameters
    # ------------------------------------------------------------------
    integral_x = 40
    integral_y = 40
    integral_z = 16

    # ------------------------------------------------------------------
    # Geometry and thickness parameters
    # ------------------------------------------------------------------
    a = 1.0
    b = 1.0
    h = a / 10

    # Power-law index (FGM)
    p = 4

    # ------------------------------------------------------------------
    # Mid-surface domain
    # ------------------------------------------------------------------
    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    # ------------------------------------------------------------------
    # Material properties (ceramic–metal FGM)
    # ------------------------------------------------------------------
    E_M, nu_M, rho_M = 70e9, 0.30, 2710
    E_C, nu_C, rho_C = 380e9, 0.30, 3800

    Vc = lambda z: (0.5 + z / h) ** p

    material = FunctionallyGradedMaterial(
        E_C, E_M,
        nu_C, nu_M,
        rho_C, rho_M,
        Vc,
    )

    # ------------------------------------------------------------------
    # Kinematic approximation (FSDT5)
    # ------------------------------------------------------------------
    n_modes = 15

    expansion_size = {
        "u1": (n_modes, n_modes),
        "u2": (n_modes, n_modes),
        "u3": (n_modes, n_modes),
        "v1": (n_modes, n_modes),
        "v2": (n_modes, n_modes),
    }

    displacement_field = EnrichedCosineExpansion(
        expansion_size,
        rectangular_domain,
        simply_supported_fsdt5,
    )

    thickness = ConstantThickness(h)

    # ------------------------------------------------------------------
    # Geometry cases to be analyzed
    # ------------------------------------------------------------------
    geometry_cases = {
        "plate": {"case": "plate", "Rx": None, "Ry": None},
        "sphere_A": {"case": "doubly_curved", "Rx": 2, "Ry": 2},
        "sphere_B": {"case": "doubly_curved", "Rx": 1, "Ry": 1},
        "hyperbolic_A": {"case": "doubly_curved", "Rx": 2, "Ry": -2},
        "hyperbolic_B": {"case": "doubly_curved", "Rx": 1, "Ry": -1},
        "cylindrical_A": {"case": "cylindrical_x", "Rx": 2, "Ry": None},
        "cylindrical_B": {"case": "cylindrical_x", "Rx": 1, "Ry": None},
    }

    shell = None

    # ------------------------------------------------------------------
    # Results container
    # ------------------------------------------------------------------
    results = {}

    # ------------------------------------------------------------------
    # Loop over geometry cases
    # ------------------------------------------------------------------
    for name, params in geometry_cases.items():

        print(f"\nRunning case: {name}")

        R_ = midsurface_geometry(
            case=params["case"],
            a=a,
            b=b,
            Rx=params["Rx"],
            Ry=params["Ry"],
        )

        mid_surface_geometry = MidSurfaceGeometry(R_)

        clear_cache(shell)

        shell = Shell(
            mid_surface_geometry,
            thickness,
            rectangular_domain,
            material,
            displacement_field,
            load=None,
        )

        # --------------------------------------------------------------
        # Energy expressions
        # --------------------------------------------------------------
        T = kinetic_energy(shell, integral_x, integral_y, integral_z)
        U2 = quadratic_strain_energy(shell, integral_x, integral_y, integral_z)

        # --------------------------------------------------------------
        # Mass and stiffness matrices
        # --------------------------------------------------------------
        M = tensor_derivative(tensor_derivative(T, 0), 1)
        K = tensor_derivative(tensor_derivative(U2, 0), 1)

        # --------------------------------------------------------------
        # Generalized eigenvalue problem
        # --------------------------------------------------------------
        eigen_vals, eigen_vectors = eig(K, M)

        # ------------------------------------------------------------------
        # Keep only positive eigenvalues (physical modes)
        # ------------------------------------------------------------------
        tol = 1e-8
        positive_mask = eigen_vals > tol

        eigen_vals = eigen_vals[positive_mask]
        eigen_vectors = eigen_vectors[:, positive_mask]

        # ------------------------------------------------------------------
        # Natural frequencies
        # ------------------------------------------------------------------
        omega = np.real(np.sqrt(eigen_vals))

        # ------------------------------------------------------------------
        # Sort frequencies in ascending order
        # ------------------------------------------------------------------
        idx = np.argsort(omega)
        omega = omega[idx]
        eigen_vectors = np.real(eigen_vectors[:, idx])

        # --------------------------------------------------------------
        # Frequencies
        # --------------------------------------------------------------
        freq_normalized = omega * h * np.sqrt(rho_C / E_C)
        freq_hz = omega / (2.0 * np.pi)

        results[name] = {
            "omega": omega,
            "freq_normalized": freq_normalized,
            "freq_hz": freq_hz,
            "eigen_vectors": eigen_vectors,
            "shell": shell,
        }

        # Print only the first (fundamental) mode for each case
        print("First normalized frequency:", freq_normalized[0])
    # ------------------------------------------------------------------
    # Final post-processing: print all frequencies and plot first mode
    # ------------------------------------------------------------------
    print("================= SUMMARY OF RESULTS =================")

    for name, data in results.items():
        print(f"Case: {name}")
        print("Normalized frequencies:")
        print(data["freq_normalized"][0:5])

        # Plot only mode shape
        shell = data["shell"]

        for mode in range(5):
            phi1 = data["eigen_vectors"][:, mode]
            file_name = f"{name}_mode_{mode}.png"
            shell_mode(
                shell,
                phi1,
                file_name,
                n_1=40,
                n_2=40,
                n_3=4,
                max_deformation=0.5 * h,
            )
