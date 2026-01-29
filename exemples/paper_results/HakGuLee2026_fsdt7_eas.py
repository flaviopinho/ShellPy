#Refined shear correction factors for composite-layered FE shell elements to
#enhance the accuracy of their modal analysis results
#Hak Gu Lee a,*
#, Dong-KuK Choi a
#, Daeyong Kwon b,*
#, Semyung Park

# Table 6

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
    integral_x = 40
    integral_y = 40
    integral_z = 10

    a = 1
    b = 1
    h = 4 * 12.5e-3

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    R_ = sym.Matrix([xi1_, xi2_, 0])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)

    density = 1500

    E1 = 103.421e9
    E2 = 6.89476e9
    E3 = 6.89476e9

    nu12 = 0.3
    nu13 = 0.3
    nu23 = 0.42

    G12 = 3.44738e9
    G13 = 3.44738e9
    G23 = 2.41317e9

    lamina10p = Lamina(
        E_11=E1,  # Pa
        E_22=E2,
        E_33=E3,
        nu_12=nu12,
        nu_13=nu13,
        nu_23=nu23,
        G_12=G12,
        G_13=G13,
        G_23=G23,
        density=density,
        angle=np.deg2rad(10),
        thickness=2 / 5
    )

    lamina5p = Lamina(
        E_11=E1,  # Pa
        E_22=E2,
        E_33=E3,
        nu_12=nu12,
        nu_13=nu13,
        nu_23=nu23,
        G_12=G12,
        G_13=G13,
        G_23=G23,
        density=density,  # kg/m³
        angle=np.deg2rad(5),  # orientação (graus ou rad, depende da convenção)
        thickness=2 / 5  # 1 mm
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
        density=density,  # kg/m³
        angle=0,  # orientação (graus ou rad, depende da convenção)
        thickness=2 / 5  # 1 mm
    )

    lamina5m = Lamina(
        E_11=E1,  # Pa
        E_22=E2,
        E_33=E3,
        nu_12=nu12,
        nu_13=nu13,
        nu_23=nu23,
        G_12=G12,
        G_13=G13,
        G_23=G23,
        density=density,
        angle=-np.deg2rad(5),
        thickness=2 / 5
    )

    lamina10m = Lamina(
        E_11=E1,  # Pa
        E_22=E2,
        E_33=E3,
        nu_12=nu12,
        nu_13=nu13,
        nu_23=nu23,
        G_12=G12,
        G_13=G13,
        G_23=G23,
        density=density,  # kg/m³
        angle=-np.deg2rad(10),  # orientação (graus ou rad, depende da convenção)
        thickness= 2 / 5  # 1 mm
    )

    material = LaminateOrthotropicMaterial(
        [lamina10p, lamina5p, lamina0, lamina5m, lamina10m], thickness)

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
    n_modes = 10

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

    # Print the first five natural frequencies
    print("Frequencies (Hz):\n", freqHz[0:n_modes:1])

    # Generate a mesh grid for visualization of mode shapes
    xi1 = np.linspace(*rectangular_domain.edges["xi1"], 100)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], 100)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)

    # Create a figure for mode shape visualization
    fig, axes = plt.subplots(1, n_modes, figsize=(15, 5), subplot_kw={'projection': '3d'}, constrained_layout=True)

    # Loop through the first few vibration modes
    for i in range(n_modes):
        mode1 = shell.displacement_expansion(eigen_vectors[:, i], x, y)  # Compute mode shape

        mode = reciprocal_base[0] * mode1[0] + reciprocal_base[1] * mode1[1] + reciprocal_base[2] * mode1[2]

        mode = mode / np.max(np.abs(mode)) * h  # Normalize and scale for visualization

        z = shell.mid_surface_geometry(x, y)  # Compute deformed geometry

        ax = axes[i]  # Select subplot
        scmap = plt.cm.ScalarMappable(cmap='jet')  # Define colormap
        ax.plot_surface(z[0, 0] + mode[0], z[1, 0] + mode[1], z[2, 0] + mode[2],
                        facecolors=scmap.to_rgba(mode1[2]),
                        edgecolor='black',
                        linewidth=0.1)  # Plot mode shape

        # Label axes and set the title with frequency information
        ax.set_title(f"Mode {i + 1} - Frequency: {freqHz[i]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Ensure equal aspect ratio for visualization
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    # Adjust layout and display the plots
    # plt.tight_layout()
    plt.show()
