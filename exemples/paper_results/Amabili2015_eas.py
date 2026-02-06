"""
Modal analysis of a cylindrical shell based on:
Amabili (2015) – Non-linearities in rotation and thickness deformation
in a third-order thickness deformation theory for static and dynamic
analysis of isotropic and laminated doubly curved shells.

This script reproduces (partially) the numerical results reported in
Table 1 of the reference, focusing on the *linear modal analysis* stage.

Main steps:
1. Define geometry, material, and domain
2. Define kinematic expansions and boundary conditions
3. Assemble mass and stiffness matrices
4. Solve the generalized eigenvalue problem
5. Post-process natural frequencies and mode shapes
"""

# ======================================================================
# Imports
# ======================================================================

import numpy as np
import sympy as sym
from scipy.linalg import eig
import matplotlib.pyplot as plt
import pyvista as pv

# ShellPy core imports
from shellpy import (
    Shell,
    ConstantThickness,
    RectangularMidSurfaceDomain,
    MidSurfaceGeometry,
    xi1_, xi2_,
)

# Kinematic expansions
from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy.fsdt7_eas.EAS_expansion import EasExpansion

# Finite strain EAS formulation
from shellpy.fsdt7_eas.mass_matrix import mass_matrix
from shellpy.fsdt7_eas.stiffness_matrix import stiffness_matrix

# Materials
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import (
    IsotropicHomogeneousLinearElasticMaterial,
)

# Post-processing
from exemples.paper_results.shell_mode import shell_mode


# ======================================================================
# Main execution
# ======================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Numerical integration parameters
    # ------------------------------------------------------------------
    integral_x = 20   # Gauss points along xi1
    integral_y = 20   # Gauss points along xi2
    integral_z = 10   # Through-thickness integration points

    # ------------------------------------------------------------------
    # Geometry and material parameters (SI units)
    # ------------------------------------------------------------------
    R = 0.15          # Radius (m)
    L = 0.52          # Length (m)
    h = 0.03          # Thickness (m)

    E = 198e9         # Young's modulus (Pa)
    nu = 0.30         # Poisson ratio (-)
    rho = 7800        # Density (kg/m^3)

    # ------------------------------------------------------------------
    # Mid-surface domain and geometry definition
    # ------------------------------------------------------------------
    # xi1 -> axial coordinate, xi2 -> circumferential coordinate
    rectangular_domain = RectangularMidSurfaceDomain(0, L, 0, 2 * np.pi)

    # Cylindrical mid-surface parametrization
    R_ = sym.Matrix([
        R * sym.cos(xi2_),
        R * sym.sin(xi2_),
        xi1_,
    ])

    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)

    # ------------------------------------------------------------------
    # Material model
    # ------------------------------------------------------------------
    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, rho)

    # ------------------------------------------------------------------
    # Kinematic approximation orders
    # ------------------------------------------------------------------
    n_modos = 10       # Axial modes
    n_modos_c = 10     # Circumferential modes

    expansion_size = {
        "u1": (n_modos, n_modos_c),
        "u2": (n_modos, n_modos_c),
        "u3": (n_modos, n_modos_c),
        "v1": (n_modos, n_modos_c),
        "v2": (n_modos, n_modos_c),
        "v3": (n_modos, n_modos_c),
    }

    # ------------------------------------------------------------------
    # Boundary conditions
    # F = Free, S = Simply supported, R = Periodic (circumferential)
    # ------------------------------------------------------------------
    boundary_conditions = {
        "u1": {"xi1": ("F", "F"), "xi2": ("R", "R")},
        "u2": {"xi1": ("S", "S"), "xi2": ("R", "R")},
        "u3": {"xi1": ("S", "S"), "xi2": ("R", "R")},
        "v1": {"xi1": ("F", "F"), "xi2": ("R", "R")},
        "v2": {"xi1": ("S", "S"), "xi2": ("R", "R")},
        "v3": {"xi1": ("S", "S"), "xi2": ("R", "R")},
    }

    # ------------------------------------------------------------------
    # Displacement and EAS fields
    # ------------------------------------------------------------------
    displacement_field = EnrichedCosineExpansion(
        expansion_size,
        rectangular_domain,
        boundary_conditions,
    )

    eas_field = EasExpansion(
        {"eas": (n_modos, n_modos_c)},
        rectangular_domain,
        {"eas": {"xi1": ("F", "F"), "xi2": ("R", "R")}},
    )

    # ------------------------------------------------------------------
    # Shell model assembly
    # ------------------------------------------------------------------
    shell = Shell(
        mid_surface_geometry,
        thickness,
        rectangular_domain,
        material,
        displacement_field,
        load=None,
    )

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # ------------------------------------------------------------------
    # Mass and stiffness matrices
    # ------------------------------------------------------------------
    M = mass_matrix(shell, integral_x, integral_y, integral_z)
    K = stiffness_matrix(shell, eas_field, integral_x, integral_y, integral_z)

    # ------------------------------------------------------------------
    # Generalized eigenvalue problem: K φ = ω² M φ
    # ------------------------------------------------------------------
    eigen_vals, eigen_vectors = eig(K, M)
    omega = np.sqrt(eigen_vals)

    # ------------------------------------------------------------------
    # Filtering spurious / non-physical eigenvalues
    # ------------------------------------------------------------------
    tolerance = 1e-2

    finite_mask = np.isfinite(omega)
    real_nonzero_mask = np.abs(np.real(omega)) > tolerance

    valid_mask = finite_mask & real_nonzero_mask

    omega = np.real(omega[valid_mask])
    eigen_vectors = np.real(eigen_vectors[:, valid_mask])

    # ------------------------------------------------------------------
    # Sorting modes by ascending frequency
    # ------------------------------------------------------------------
    sorted_indices = np.argsort(omega)
    omega = omega[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]

    # ------------------------------------------------------------------
    # Natural frequencies (Hz)
    # ------------------------------------------------------------------
    freqHz = omega / (2.0 * np.pi)

    # Selected modes (as in the reference paper)
    idx = np.array([0, 2, 4, 17, 46, 94, 288, 325, 506])

    print("Natural frequencies (Hz):")
    print(freqHz[idx])

    # ------------------------------------------------------------------
    # Mode shape visualization
    # ------------------------------------------------------------------
    for mode_id in idx:
        file_name = f"amabili_{mode_id}.png"
        shell_mode(
            shell,
            eigen_vectors[:, mode_id],
            file_name,
            n_1=40,
            n_2=80,
            n_3=4,
            max_deformation=0.5 * h,
        )
