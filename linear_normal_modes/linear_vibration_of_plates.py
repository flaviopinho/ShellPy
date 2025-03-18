"""
This example demonstrates how to determine the natural frequencies and vibration modes of a plate.
This plate was previously studied by Boumediene et al. (2009) (see https://doi.org/10.1016/j.compstruc.2009.07.005)
"""

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

from shellpy import pinned, fully_clamped
from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy.koiter_shell_theory import fast_koiter_quadratic_strain_energy
from shellpy.koiter_shell_theory.fast_koiter_kinetic_energy import fast_koiter_kinetic_energy
from shellpy import LinearElasticMaterial
from shellpy import RectangularMidSurfaceDomain
from shellpy import xi1_, xi2_, MidSurfaceGeometry
from shellpy import Shell
from shellpy.tensor_derivatives import tensor_derivative
from shellpy import ConstantThickness

# Main execution block
if __name__ == "__main__":

    # Define geometric parameters of the shell
    a = 0.6
    b = 0.3
    h = 0.001
    density = 2778

    # Define material properties
    E = 70E9  # Young's modulus
    nu = 0.3  # Poisson’s ratio

    n_int_x = 30
    n_int_y = 30

    # Define the rectangular mid-surface domain of the shell
    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    # Define the number of terms used in the displacement expansion
    expansion_size = {"u1": (0, 0),  # Expansion order for displacement u1
                      "u2": (0, 0),  # Expansion order for displacement u2
                      "u3": (15, 15)}  # Expansion order for displacement u3

    # Define boundary conditions (pinned edges)
    boundary_conditions = fully_clamped

    # Define the displacement field using an enriched cosine expansion
    displacement_field = EnrichedCosineExpansion(expansion_size, rectangular_domain, boundary_conditions)

    # Define the symbolic representation of the mid-surface geometry
    # The surface is assumed to be a portion of a sphere
    R_ = sym.Matrix([xi1_, xi2_, 0])

    # Create objects representing the shell geometry, thickness, and material properties
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = LinearElasticMaterial(E, nu, density)

    # Instantiate the shell object with all the defined properties
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, None)

    # Determine the number of degrees of freedom in the displacement field
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    T = fast_koiter_kinetic_energy(shell, n_int_x, n_int_y)

    U2p = fast_koiter_quadratic_strain_energy(shell, n_int_x, n_int_y)

    # Compute the mass (M) and stiffness (K) matrices
    M = tensor_derivative(tensor_derivative(T, 0), 1)  # Second derivative of kinetic energy (mass matrix)
    K = tensor_derivative(tensor_derivative(U2p, 0), 1)  # Second derivative of strain energy (stiffness matrix)

    # Solve the eigenvalue problem for natural frequencies and mode shapes
    eigen_vals, eigen_vectors = eig(K, M, right=True, left=False)
    sorted_indices = np.argsort(eigen_vals.real)  # Sort eigenvalues in ascending order

    # Extract sorted eigenvalues and eigenvectors
    eigen_vals = eigen_vals[sorted_indices]
    eigen_vectors = np.real(eigen_vectors[:, sorted_indices])

    # Compute natural frequencies (Hz)
    omega = np.sqrt(eigen_vals.real)

    freq = omega

    # Number of modes to be analyzed
    n_modes = 5

    # Print the first five natural frequencies
    print("Natural frequencies (rad/s):", freq[:n_modes])

    # Generate a mesh grid for visualization of mode shapes
    xi1 = np.linspace(*rectangular_domain.edges["xi1"], 50)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], 25)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)

    # Create a figure for mode shape visualization
    fig, axes = plt.subplots(1, n_modes, figsize=(15, 5), subplot_kw={'projection': '3d'})

    # Loop through the first few vibration modes
    for i in range(n_modes):
        mode1 = shell.displacement_expansion(eigen_vectors[:, i], x, y)  # Compute mode shape

        mode = reciprocal_base[0] * mode1[0] + reciprocal_base[1] * mode1[1] + reciprocal_base[2] * mode1[2]
        mode = mode / np.max(np.abs(mode)) * h * 20  # Normalize and scale for visualization
        z = shell.mid_surface_geometry(x, y)  # Compute deformed geometry

        ax = axes[i]  # Select subplot
        scmap = plt.cm.ScalarMappable(cmap='jet')  # Define colormap
        ax.plot_surface(z[0, 0] + mode[0], z[1, 0] + mode[1], z[2, 0] + mode[2],
                        facecolors=scmap.to_rgba(mode1[2]),
                        edgecolor='black',
                        linewidth=0.1)  # Plot mode shape

        # Label axes and set the title with frequency information
        ax.set_title(f"Mode {i + 1} - Frequency: {freq[i]:.2f} rad/s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Ensure equal aspect ratio for visualization
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
