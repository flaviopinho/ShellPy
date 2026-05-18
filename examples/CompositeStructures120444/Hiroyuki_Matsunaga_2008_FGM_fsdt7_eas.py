"""
Free vibration analysis of functionally graded plates and shells
based on a First-Order Shear Deformation Theory with Enhanced Assumed Strains
(FSDT6/FSDT7 + EAS).

This script performs a linear free-vibration analysis of functionally graded
plates, cylindrical shells and doubly curved shells using the ShellPy framework.
The formulation combines an enriched trigonometric kinematic expansion with
an EAS approach to improve transverse shear and membrane behavior.

The code is structured to automatically run multiple geometric curvature cases
(flat plate, cylindrical shell and doubly curved shell) using a unified
solver pipeline.

Main steps:
1. Definition of mid-surface geometry and curvature cases
2. Functionally graded material modeling (power-law distribution)
3. Kinematic approximation using enriched cosine expansions (FSDT)
4. Assembly of mass and stiffness matrices via numerical integration
5. Solution of the generalized eigenvalue problem
6. Extraction of normalized and dimensional natural frequencies
7. Post-processing and visualization of vibration mode shapes
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
)
from shellpy.displacement_expansion import simply_supported_fsdt6
from shellpy.cache_decorator import clear_cache

# Expansions
from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy.fsdt7_eas.EAS_expansion import EasExpansion

# Matrices
from shellpy.fsdt7_eas.mass_matrix import mass_matrix
from shellpy.fsdt7_eas.stiffness_matrix import stiffness_matrix

# Materials
from shellpy.materials.functionally_graded_material import FunctionallyGradedMaterial

# Plot utility
from shellpy.utils.shell_mode import shell_mode


# ======================================================================
# Geometry factory
# ======================================================================

def midsurface_geometry(case, a, b, Rx=None, Ry=None):
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
    integral_x = 20
    integral_y = 20
    integral_z = 16

    # ------------------------------------------------------------------
    # Geometry and thickness
    # ------------------------------------------------------------------
    a = 1.0
    b = 1.0
    h = a / 10

    # ------------------------------------------------------------------
    # Power-law index (FGM)
    # ------------------------------------------------------------------
    p = 4

    # ------------------------------------------------------------------
    # Mid-surface domain
    # ------------------------------------------------------------------
    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    # ------------------------------------------------------------------
    # Material properties
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
    # Kinematic approximation (FSDT6 + EAS)
    # ------------------------------------------------------------------
    n_modes = 15

    expansion_size = {
        "u1": (n_modes, n_modes),
        "u2": (n_modes, n_modes),
        "u3": (n_modes, n_modes),
        "v1": (n_modes, n_modes),
        "v2": (n_modes, n_modes),
        "v3": (n_modes, n_modes),
    }

    displacement_field = EnrichedCosineExpansion(
        expansion_size,
        rectangular_domain,
        simply_supported_fsdt6,
    )

    eas_field = EasExpansion(
        {"eas": (n_modes, n_modes)},
        rectangular_domain,
        {"eas": {"xi1": ("F", "F"), "xi2": ("F", "F")}},
    )

    thickness = ConstantThickness(h)

    # ------------------------------------------------------------------
    # Geometry cases
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
            None,
        )

        # --------------------------------------------------------------
        # Mass and stiffness matrices
        # --------------------------------------------------------------
        M = mass_matrix(shell, integral_x, integral_y, integral_z)
        K = stiffness_matrix(shell, eas_field, integral_x, integral_y, integral_z)

        # --------------------------------------------------------------
        # Generalized eigenvalue problem
        # --------------------------------------------------------------
        eigen_vals, eigen_vectors = eig(K, M)
        omega = np.sqrt(eigen_vals)

        # --------------------------------------------------------------
        # Filtering (physical modes only)
        # --------------------------------------------------------------
        tol = 1e-2
        mask = np.isfinite(omega) & (np.abs(np.real(omega)) > tol)

        omega = np.real(omega[mask])
        eigen_vectors = np.real(eigen_vectors[:, mask])

        # --------------------------------------------------------------
        # Sorting
        # --------------------------------------------------------------
        idx = np.argsort(omega)
        omega = omega[idx]
        eigen_vectors = eigen_vectors[:, idx]

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

        # --------------------------------------------------------------
        # Print only first mode (as requested)
        # --------------------------------------------------------------
        print("First normalized frequency:", freq_normalized[0])

    # ==================================================================
    # Final post-processing
    # ==================================================================

    print("\n================= SUMMARY OF RESULTS =================")

    for name, data in results.items():
        print(f"\nCase: {name}")
        print("Normalized frequencies:")
        print(data["freq_normalized"][:5])

    # ==================================================================
    # Mode-shape plots (shell_mode)
    # ==================================================================

    for name, data in results.items():

        shell = data["shell"]
        eigen_vectors = data["eigen_vectors"]

        # Plot only first mode (change range if needed)
        for mode_id in [0]:
            file_name = f"{name}_mode_{mode_id}.png"

            shell_mode(
                shell,
                eigen_vectors[:, mode_id],
                file_name,
                n_1=40,
                n_2=80,
                n_3=4,
                max_deformation=0.5 * h,
            )
