#Non-linearities in rotation and thickness deformation in a new
#third-order thickness deformation theory for static and dynamic
#analysis of isotropic and laminated doubly curved shells
#Amabili 2015

# Table 1

import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
from scipy.linalg import eig

from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy.fsdt5.kinetic_energy import kinetic_energy
from shellpy.fsdt5.strain_energy import quadratic_strain_energy
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.materials.laminate_orthotropic_material import Lamina, LaminateOrthotropicMaterial
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_
from shellpy.tensor_derivatives import tensor_derivative

if __name__ == "__main__":
    integral_x = 20
    integral_y = 20
    integral_z = 10

    R = 0.15
    L = 0.52
    h = 0.015
    E = 198E9
    nu = 0.3
    rho = 7800

    rectangular_domain = RectangularMidSurfaceDomain(0, L, 0, 2 * np.pi)

    R_ = sym.Matrix([R * sym.cos(xi2_), R * sym.sin(xi2_), xi1_])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)

    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, rho)

    n_modos_c = 10
    n_modos = 10
    expansion_size = {"u1": (n_modos, n_modos_c),
                      "u2": (n_modos, n_modos_c),
                      "u3": (n_modos, n_modos_c),
                      "v1": (n_modos, n_modos_c),
                      "v2": (n_modos, n_modos_c),
                      "v3": (n_modos, n_modos_c)}

    boundary_conditions_u1 = {"xi1": ("F", "F"),
                              "xi2": ("R", "R")}
    boundary_conditions_u2 = {"xi1": ("S", "S"),
                              "xi2": ("R", "R")}
    boundary_conditions_u3 = {"xi1": ("S", "S"),
                              "xi2": ("R", "R")}

    boundary_conditions_v1 = {"xi1": ("F", "F"),
                              "xi2": ("R", "R")}
    boundary_conditions_v2 = {"xi1": ("F", "F"),
                              "xi2": ("R", "R")}
    boundary_conditions_v3 = {"xi1": ("F", "F"),
                              "xi2": ("R", "R")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3,
                           "v1": boundary_conditions_v1,
                           "v2": boundary_conditions_v2,
                           "v3": boundary_conditions_v3}

    displacement_field = EnrichedCosineExpansion(expansion_size, rectangular_domain, boundary_conditions)

    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, None)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    T = kinetic_energy(shell, integral_x, integral_y, integral_z)
    U2p = quadratic_strain_energy(shell, integral_x, integral_y, integral_z)

    # Compute the mass (M) and stiffness (K) matrices
    M = tensor_derivative(tensor_derivative(T, 0), 1)  # Second derivative of kinetic energy (mass matrix)
    K = tensor_derivative(tensor_derivative(U2p, 0), 1)  # Second derivative of strain energy (stiffness matrix)

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
