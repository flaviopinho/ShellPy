import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, eigh

from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy.koiter_shell_theory import fast_koiter_quadratic_strain_energy
from shellpy.koiter_shell_theory.fast_koiter_kinetic_energy import fast_koiter_kinetic_energy
from shellpy import RectangularMidSurfaceDomain
from shellpy import xi1_, xi2_, MidSurfaceGeometry
from shellpy import Shell
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.tensor_derivatives import tensor_derivative
from shellpy import ConstantThickness

# Main execution block
if __name__ == "__main__":

    n_int_x = 40
    n_int_y = 40
    n_int_z = 4

    factor = 0.5
    alpha = np.deg2rad(30)
    R2 = 0.4
    L = R2 * factor / np.sin(alpha)
    L2 = R2 / np.sin(alpha)
    L1 = L2 - L
    R1 = L1 * np.sin(alpha)

    H = 0.01 * R2
    p = 1
    q = 1


    def thickness(xi1, xi2):
        zeta0 = 0.5
        s1 = 0.5
        s2 = 1

        threshold = L1 + zeta0 * L
        h_values = np.where(
            (xi1 >= L1) & (xi1 <= threshold),
            s1 * H,
            s2 * H
        )
        return h_values


    rectangular_domain = RectangularMidSurfaceDomain(L1, L2, 0, 2 * np.pi)

    R_ = sym.Matrix([
        xi1_ * sym.sin(alpha) * sym.cos(xi2_),  # x
        xi1_ * sym.sin(alpha) * sym.sin(xi2_),  # y
        xi1_ * sym.cos(alpha)  # z
    ])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    density = 2710

    E = 70E9
    nu = 0.3

    # Define the number of terms used in the displacement expansion
    expansion_size = {"u1": (20, 20),  # Expansion order for displacement u1
                      "u2": (20, 20),  # Expansion order for displacement u2
                      "u3": (20, 20)}  # Expansion order for displacement u3

    # Define boundary conditions
    boundary_conditions_u1 = {"xi1": ("S", "S"),
                              "xi2": ("R", "R")}
    boundary_conditions_u2 = {"xi1": ("S", "S"),
                              "xi2": ("R", "R")}
    boundary_conditions_u3 = {"xi1": ("C", "C"),
                              "xi2": ("R", "R")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3}

    # Define the displacement field using an enriched cosine expansion
    displacement_field = EigenFunctionExpansion(expansion_size, rectangular_domain, boundary_conditions)

    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, density)

    # Instantiate the shell object with all the defined properties
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, None)

    # Determine the number of degrees of freedom in the displacement field
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    T = fast_koiter_kinetic_energy(shell, n_int_x, n_int_y, n_int_z)

    U2p = fast_koiter_quadratic_strain_energy(shell, n_int_x, n_int_y, n_int_z)

    # Compute the mass (M) and stiffness (K) matrices
    M = tensor_derivative(tensor_derivative(T, 0), 1)  # Second derivative of kinetic energy (mass matrix)
    K = tensor_derivative(tensor_derivative(U2p, 0), 1)  # Second derivative of strain energy (stiffness matrix)

    # Number of modes to be analyzed
    n_modes = 30

    # Solve the eigenvalue problem for natural frequencies and mode shapes
    eigen_vals, eigen_vectors = eigh(K, M, subset_by_index=[0, n_modes - 1])
    sorted_indices = np.argsort(eigen_vals.real)  # Sort eigenvalues in ascending order

    # Extract sorted eigenvalues and eigenvectors
    eigen_vals = eigen_vals[sorted_indices]
    eigen_vectors = np.real(eigen_vectors[:, sorted_indices])

    # Compute natural frequencies (Hz)
    omega = np.sqrt(eigen_vals.real)

    freq = omega * R2 * np.sqrt(density * (1 - nu ** 2) / E)
    freqHz = omega / (2 * np.pi)

    # Print the first five natural frequencies
    print("Normalized natural frequencies:\n", freq[0:n_modes:1])
    print("Frequencies (hz):\n", freqHz[0:n_modes:1])

    # Generate a mesh grid for visualization of mode shapes
    xi1 = np.linspace(*rectangular_domain.edges["xi1"], 100)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], 50)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    # Create a figure for mode shape visualization
    fig, axes = plt.subplots(1, n_modes // 2, figsize=(20, 5), subplot_kw={'projection': '3d'}, constrained_layout=True)

    # Loop through the first few vibration modes

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)

    for i in range(0, 0, 2):
        mode1 = shell.displacement_expansion(eigen_vectors[:, i], x, y)  # Compute mode shape

        mode = reciprocal_base[0] * mode1[0] + reciprocal_base[1] * mode1[1] + reciprocal_base[2] * mode1[2]

        mode = mode / np.max(np.abs(mode)) * 0.02  # Normalize and scale for visualization

        z = shell.mid_surface_geometry(x, y)  # Compute deformed geometry

        ax = axes[i // 2]  # Select subplot
        scmap = plt.cm.ScalarMappable(cmap='jet')  # Define colormap
        ax.plot_surface(z[0, 0] + mode[0], z[1, 0] + mode[1], z[2, 0] + mode[2],
                        facecolors=scmap.to_rgba(mode1[2]),
                        edgecolor='black',
                        linewidth=0.1, rstride=1, cstride=1)  # Plot mode shape

        # Label axes and set the title with frequency information
        ax.set_title(f"Mode {i + 1} - Frequency: {freq[i]:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Ensure equal aspect ratio for visualization
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    # Adjust layout and display the plots
    # plt.tight_layout()
    plt.show()
