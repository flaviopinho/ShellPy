# Table 6.12

import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
from scipy.linalg import eig

from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy.expansions.polinomial_expansion import LegendreSeries
from shellpy.fsdt7_eas.mass_matrix import mass_matrix
from shellpy.fsdt7_eas.stiffness_matrix import stiffness_matrix
from shellpy.materials.laminate_orthotropic_material import Lamina, LaminateOrthotropicMaterial
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_

if __name__ == "__main__":
    integral_x = 20
    integral_y = 20
    integral_z = 8

    R = 127.5E-3
    a = 152.4E-3
    b = 76.2E-3
    h = 8 * 0.13E-3

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 + -(xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)

    density = 1500

    E1 = 128E9
    E2 = 11E9
    E3 = 11E9

    nu12 = 0.25
    nu13 = 0.25
    nu23 = 0.45

    G12 = 4.48E9
    G13 = 4.48E9
    G23 = 1.53E9

    lamina45p = Lamina(
        E_11=E1,  # Pa
        E_22=E2,
        E_33=E3,
        nu_12=nu12,
        nu_13=nu13,
        nu_23=nu23,
        G_12=G12,
        G_13=G13,
        G_23=G23,
        density=1500,
        angle=np.pi/4,
        thickness=1/4
    )

    lamina45m = Lamina(
        E_11=E1,  # Pa
        E_22=E2,
        E_33=E3,
        nu_12=nu12,
        nu_13=nu13,
        nu_23=nu23,
        G_12=G12,
        G_13=G13,
        G_23=G23,
        density=1500,  # kg/m³
        angle=-np.pi/4,  # orientação (graus ou rad, depende da convenção)
        thickness=1/4  # 1 mm
    )

    lamina0 = Lamina(
        E_11=E1,  # Pa
        E_22=E2,
        E_33=E3,
        nu_12=nu12,
        nu_13=nu13,
        nu_23=nu23,
        G_12=G12,
        G_13=G13,
        G_23=G23,
        density=1500,  # kg/m³
        angle=0,  # orientação (graus ou rad, depende da convenção)
        thickness=1 / 4  # 1 mm
    )

    lamina30p = Lamina(
        E_11=E1,  # Pa
        E_22=E2,
        E_33=E3,
        nu_12=nu12,
        nu_13=nu13,
        nu_23=nu23,
        G_12=G12,
        G_13=G13,
        G_23=G23,
        density=1500,
        angle=np.pi / 6,
        thickness=1 / 4
    )

    lamina30m = Lamina(
        E_11=E1,  # Pa
        E_22=E2,
        E_33=E3,
        nu_12=nu12,
        nu_13=nu13,
        nu_23=nu23,
        G_12=G12,
        G_13=G13,
        G_23=G23,
        density=1500,  # kg/m³
        angle=-np.pi / 6,  # orientação (graus ou rad, depende da convenção)
        thickness= 1 / 4  # 1 mm
    )

    lamina90 = Lamina(
        E_11=E1,  # Pa
        E_22=E2,
        E_33=E3,
        nu_12=nu12,
        nu_13=nu13,
        nu_23=nu23,
        G_12=G12,
        G_13=G13,
        G_23=G23,
        density=1500,  # kg/m³
        angle=np.pi,  # orientação (graus ou rad, depende da convenção)
        thickness=1 / 4  # 1 mm
    )

    #material = LaminateOrthotropicMaterial([lamina45m, lamina45p, lamina45p, lamina45m, lamina45m, lamina45p, lamina45p, lamina45m], thickness)
    #material = LaminateOrthotropicMaterial(
    #    [lamina0, lamina0, lamina30p, lamina30m, lamina30m, lamina30p, lamina0, lamina0], thickness)
    material = LaminateOrthotropicMaterial(
        [lamina0, lamina45m, lamina45p, lamina90, lamina90, lamina45p, lamina45m, lamina0], thickness)

    n_modos = 15
    expansion_size = {"u1": (n_modos, n_modos),
                      "u2": (n_modos, n_modos),
                      "u3": (n_modos, n_modos),
                      "v1": (n_modos, n_modos),
                      "v2": (n_modos, n_modos),
                      "v3": (n_modos, n_modos)}

    boundary_conditions_u1 = {"xi1": ("S", "F"),
                              "xi2": ("F", "F")}
    boundary_conditions_u2 = {"xi1": ("S", "F"),
                              "xi2": ("F", "F")}
    boundary_conditions_u3 = {"xi1": ("C", "F"),
                              "xi2": ("F", "F")}

    boundary_conditions_v1 = {"xi1": ("S", "F"),
                              "xi2": ("F", "F")}
    boundary_conditions_v2 = {"xi1": ("S", "F"),
                              "xi2": ("F", "F")}
    boundary_conditions_v3 = {"xi1": ("S", "F"),
                              "xi2": ("F", "F")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3,
                           "v1": boundary_conditions_v1,
                           "v2": boundary_conditions_v2,
                           "v3": boundary_conditions_v3}

    displacement_field = EnrichedCosineExpansion(expansion_size, rectangular_domain, boundary_conditions)

    eas_field = LegendreSeries({"u1": (n_modos, n_modos)},
                                        rectangular_domain,
                                        {"u1": {"xi1": ("F", "F"), "xi2": ("F", "F")}})

    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, None)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    M = mass_matrix(shell, integral_x, integral_y, integral_z)
    K = stiffness_matrix(shell, eas_field, integral_x, integral_y, integral_z)

    # Number of modes to be analyzed
    n_modes = 5

    # Solve generalized eigenvalue problem
    eigen_vals, eigen_vectors = eig(K, M)
    omega = np.sqrt(eigen_vals)

    # Keep only finite eigenvalues (remove NaN or Inf)
    finite_mask = np.isfinite(omega)

    tolerance = 1e-2
    real_part_non_zero_mask = np.abs(np.real(omega)) > tolerance

    final_mask = finite_mask & real_part_non_zero_mask
    omega = omega[final_mask]

    eigen_vectors = eigen_vectors[:, final_mask]

    # Sort eigenvalues in ascending order
    sorted_indices = np.argsort(omega.real)

    # Extract sorted finite eigenvalues and corresponding eigenvectors
    omega = omega[sorted_indices].real
    eigen_vectors = np.real(eigen_vectors[:, sorted_indices])

    # Compute natural frequencies (Hz)
    freqHz = omega / (2.0 * np.pi)

    print(freqHz)

