"""
Free vibration analysis of cantilevered doubly-curved laminated composite shells
based on:

M. S. Qatu, A. W. Leissa,
"Natural frequencies for cantilevered doubly-curved laminated composite
shallow shells",
Composite Structures 17 (3) (1991) 227–255.

This script reproduces the numerical framework used to compare different
lamination schemes for the same shell geometry and boundary conditions.
The code is structured to automatically run multiple stacking sequences
using a unified solver pipeline.

Main steps:
1. Definition of shell geometry and boundary conditions
2. Orthotropic lamina and laminate modeling
3. Kinematic approximation using enriched cosine expansions (FSDT + EAS)
4. Assembly of mass and stiffness matrices
5. Solution of the generalized eigenvalue problem
6. Extraction of natural frequencies for comparison with reference results
"""

# ======================================================================
# Imports
# ======================================================================

import sympy as sym
import numpy as np
from scipy.linalg import eig

from exemples.paper_results.shell_mode import shell_mode
from shellpy import (
    Shell,
    ConstantThickness,
    RectangularMidSurfaceDomain,
    MidSurfaceGeometry,
    xi1_, xi2_,
)
from shellpy.cache_decorator import clear_cache

from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy.expansions.polinomial_expansion import LegendreSeries
from shellpy.fsdt7_eas.EAS_expansion import EasExpansion
from shellpy.fsdt7_eas.mass_matrix import mass_matrix
from shellpy.fsdt7_eas.stiffness_matrix import stiffness_matrix
from shellpy.materials.laminate_orthotropic_material import (
    Lamina,
    LaminateOrthotropicMaterial,
)

# ======================================================================
# Geometry definition
# ======================================================================

def midsurface_geometry(R, b):
    """
    Cylindrical shallow shell geometry used by Qatu & Leissa.
    """
    return sym.Matrix([
        xi1_,
        xi2_,
        sym.sqrt(R**2 - (xi2_ - b / 2)**2),
    ])

# ======================================================================
# Lamina factory
# ======================================================================

def orthotropic_lamina(angle, thickness):
    return Lamina(
        E_11=128e9,
        E_22=11e9,
        E_33=11e9,
        nu_12=0.25,
        nu_13=0.25,
        nu_23=0.45,
        G_12=4.48e9,
        G_13=4.48e9,
        G_23=1.53e9,
        density=1500,
        angle=angle,
        thickness=thickness,
    )

# ======================================================================
# Main execution
# ======================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Numerical integration parameters
    # ------------------------------------------------------------------
    integral_x = 20
    integral_y = 20
    integral_z = 8

    # ------------------------------------------------------------------
    # Geometry and thickness
    # ------------------------------------------------------------------
    R = 127.5e-3
    a = 152.4e-3
    b = 76.2e-3
    h = 8 * 0.13e-3

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)
    mid_surface_geometry = MidSurfaceGeometry(midsurface_geometry(R, b))
    thickness = ConstantThickness(h)

    # ------------------------------------------------------------------
    # Lamina definitions
    # ------------------------------------------------------------------
    t_ply = 1 / 4

    lamina_0   = orthotropic_lamina(0.0, t_ply)
    lamina_45p = orthotropic_lamina(+np.pi / 4, t_ply)
    lamina_45m = orthotropic_lamina(-np.pi / 4, t_ply)
    lamina_30p = orthotropic_lamina(+np.pi / 6, t_ply)
    lamina_30m = orthotropic_lamina(-np.pi / 6, t_ply)
    lamina_90  = orthotropic_lamina(np.pi / 2, t_ply)

    # ------------------------------------------------------------------
    # Lamination schemes (Table 9 – Qatu & Leissa)
    # ------------------------------------------------------------------
    lamination_cases = {
        "case_A": [lamina_45p, lamina_45m, lamina_45m, lamina_45p,
                   lamina_45p, lamina_45m, lamina_45m, lamina_45p],

        "case_B": [lamina_0, lamina_0, lamina_30p, lamina_30m,
                   lamina_30m, lamina_30p, lamina_0, lamina_0],

        "case_C": [lamina_0, lamina_45p, lamina_45m, lamina_90,
                   lamina_90, lamina_45m, lamina_45p, lamina_0],
    }

    # ------------------------------------------------------------------
    # Kinematic approximation
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

    boundary_conditions = {
        "u1": {"xi1": ("S", "F"), "xi2": ("F", "F")},
        "u2": {"xi1": ("S", "F"), "xi2": ("F", "F")},
        "u3": {"xi1": ("C", "F"), "xi2": ("F", "F")},
        "v1": {"xi1": ("F", "F"), "xi2": ("F", "F")},
        "v2": {"xi1": ("F", "F"), "xi2": ("F", "F")},
        "v3": {"xi1": ("F", "F"), "xi2": ("F", "F")},
    }

    displacement_field = EnrichedCosineExpansion(
        expansion_size,
        rectangular_domain,
        boundary_conditions,
    )

    eas_field = EasExpansion(
        {"eas": (n_modes, n_modes)},
        rectangular_domain,
        {"eas": {"xi1": ("F", "F"), "xi2": ("F", "F")}},
    )

    shell = None

    # ------------------------------------------------------------------
    # Results container
    # ------------------------------------------------------------------
    results = {}

    # ------------------------------------------------------------------
    # Loop over lamination schemes
    # ------------------------------------------------------------------
    for name, stacking in lamination_cases.items():

        print(f"\nRunning lamination case: {name}")

        clear_cache(shell)

        material = LaminateOrthotropicMaterial(stacking, thickness)

        shell = Shell(
            mid_surface_geometry,
            thickness,
            rectangular_domain,
            material,
            displacement_field,
            None,
        )

        # Mass and stiffness matrices
        M = mass_matrix(shell, integral_x, integral_y, integral_z)
        K = stiffness_matrix(shell, eas_field, integral_x, integral_y, integral_z)

        # --------------------------------------------------------------
        # Generalized eigenvalue problem
        # --------------------------------------------------------------
        eigen_vals, eigen_vectors = eig(K, M)

        # Keep only physical (positive, real) eigenvalues
        tol = 1e-6
        mask = np.real(eigen_vals) > tol

        eigen_vals = np.real(eigen_vals[mask])
        eigen_vectors = np.real(eigen_vectors[:, mask])

        # Natural frequencies
        omega = np.sqrt(eigen_vals)

        # Sort frequencies and corresponding modes
        idx = np.argsort(omega)
        omega = omega[idx]
        eigen_vectors = eigen_vectors[:, idx]

        freq_hz = omega / (2.0 * np.pi)

        # Store results
        results[name] = {
            "omega": omega,
            "freq_hz": freq_hz,
            "eigen_vectors": eigen_vectors,
            "shell": shell,
        }

        # Print fundamental frequency
        print(f"Fundamental frequency (Hz): {freq_hz[0]:.4f}")

    # ------------------------------------------------------------------
    # Summary and mode shape visualization
    # ------------------------------------------------------------------
    print("\n================= SUMMARY =================")

    n_plot_modes = 5

    for name, data in results.items():
        print(f"\nCase: {name}")
        print("First five frequencies (Hz):")
        print(data["freq_hz"][:n_plot_modes])

        shell = data["shell"]
        eigen_vectors = data["eigen_vectors"]

        # Plot first modes
        for mode_id in range(n_plot_modes):
            phi = eigen_vectors[:, mode_id]

            file_name = f"{name}_mode_{mode_id + 1}.png"

            shell_mode(
                shell,
                phi,
                file_name,
                n_1=21,
                n_2=21,
                n_3=3,
                max_deformation=5 * h,
            )
