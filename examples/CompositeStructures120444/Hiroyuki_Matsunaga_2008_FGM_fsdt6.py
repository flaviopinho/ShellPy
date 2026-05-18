"""
Free vibration analysis of functionally graded shallow shells
based on:

H. Matsunaga, "Free vibration and stability of functionally graded shallow
shells according to a 2D higher-order deformation theory",
Composite Structures.

FSDT6 version (6 kinematic variables).

This script reproduces the linear free-vibration results reported in
Table 2 of the reference paper and is structured to automatically run
multiple geometric curvature cases (flat, cylindrical, doubly curved).
"""

# ======================================================================
# Imports
# ======================================================================

import numpy as np
import sympy as sym
from scipy.linalg import eig

from shellpy import (
    Shell,
    ConstantThickness,
    RectangularMidSurfaceDomain,
    MidSurfaceGeometry,
    xi1_, xi2_,
)
from shellpy.cache_decorator import clear_cache
from shellpy.displacement_expansion import simply_supported_fsdt6

from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion

# FSDT6 energies
from shellpy.fsdt6.kinetic_energy import kinetic_energy
from shellpy.fsdt6.strain_energy import quadratic_strain_energy

from shellpy.materials.functionally_graded_material import FunctionallyGradedMaterial
from shellpy.tensor_derivatives import tensor_derivative

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

    integral_x = 10
    integral_y = 10
    integral_z = 8

    a = 1.0
    b = 1.0
    h = a / 10
    p = 4

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    # Material (FGM)
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
    # FSDT6 kinematics
    # ------------------------------------------------------------------
    n_modes = 10

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

    thickness = ConstantThickness(h)

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
    results = {}

    for name, params in geometry_cases.items():

        print(f"\nRunning case: {name}")

        R_ = midsurface_geometry(
            params["case"], a, b, params["Rx"], params["Ry"]
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

        T = kinetic_energy(shell, integral_x, integral_y, integral_z)
        U2 = quadratic_strain_energy(shell, integral_x, integral_y, integral_z)

        M = tensor_derivative(tensor_derivative(T, 0), 1)
        K = tensor_derivative(tensor_derivative(U2, 0), 1)

        eigen_vals, eigen_vectors = eig(K, M)

        positive_mask = eigen_vals > 1e-8
        eigen_vals = eigen_vals[positive_mask]
        eigen_vectors = eigen_vectors[:, positive_mask]

        omega = np.sqrt(np.real(eigen_vals))
        idx = np.argsort(omega)

        omega = omega[idx]
        eigen_vectors = np.real(eigen_vectors[:, idx])

        freq_normalized = omega * h * np.sqrt(rho_C / E_C)
        freq_hz = omega / (2 * np.pi)

        results[name] = {
            "omega": omega,
            "freq_normalized": freq_normalized,
            "freq_hz": freq_hz,
            "eigen_vectors": eigen_vectors,
            "shell": shell,
        }

        print("First normalized frequency:", freq_normalized[0])

    print("\n================= SUMMARY =================")

    for name, data in results.items():
        print(f"\nCase: {name}")
        print("Normalized frequencies:", data["freq_normalized"][:5])

        shell = data["shell"]

        for mode in range(5):
            shell_mode(
                shell,
                data["eigen_vectors"][:, mode],
                f"{name}_mode_{mode}.png",
                n_1=40,
                n_2=40,
                n_3=4,
                max_deformation=0.5 * h,
            )
