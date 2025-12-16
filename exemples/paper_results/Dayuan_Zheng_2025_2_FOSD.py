import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
from scipy.linalg import eig, eigh

from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy.fsdt6.kinetic_energy import kinetic_energy
from shellpy.fsdt6.strain_energy import quadratic_strain_energy
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.tensor_derivatives import tensor_derivative
from shellpy import Shell
from shellpy import MidSurfaceGeometry, xi1_, xi2_

if __name__ == "__main__":
    integral_x = 40
    integral_y = 40
    integral_z = 4

    factor = 0.5
    alpha = np.deg2rad(30)
    R2 = 0.4
    L = R2 * factor / np.sin(alpha)
    L2 = R2 / np.sin(alpha)
    L1 = L2 - L
    R1 = L1 * np.sin(alpha)

    rectangular_domain = RectangularMidSurfaceDomain(L1, L2, 0, 2 * np.pi)

    R_ = sym.Matrix([
        xi1_ * sym.sin(alpha) * sym.cos(xi2_),  # x
        xi1_ * sym.sin(alpha) * sym.sin(xi2_),  # y
        xi1_ * sym.cos(alpha)  # z
    ])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    H = 0.01 * R2


    def thickness(xi1, xi2):
        s1 = 0.5
        s2 = 1.0
        qi0 = 0.5

        cutoff = L1 + qi0 * L

        # np.where works elementwise, so it handles 2D arrays naturally
        h = np.where(xi1 <= cutoff, s1 * H, s2 * H)

        return h

    rho = 2710
    E = 1
    nu = 0.3

    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, rho)

    n_modos_1 = 20
    n_modos_2 = 22

    expansion_size = {"u1": (n_modos_1, n_modos_2),
                      "u2": (n_modos_1, n_modos_2),
                      "u3": (n_modos_1, n_modos_2),
                      "v1": (n_modos_1, n_modos_2),
                      "v2": (n_modos_1, n_modos_2),
                      "v3": (n_modos_1, n_modos_2)}

    boundary_conditions_u1 = {"xi1": ("S", "S"),
                              "xi2": ("R", "R")}
    boundary_conditions_u2 = {"xi1": ("S", "S"),
                              "xi2": ("R", "R")}
    boundary_conditions_u3 = {"xi1": ("C", "C"),
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
    n_modes = 50

    # Solve the eigenvalue problem for natural frequencies and mode shapes
    eigen_vals, eigen_vectors = eigh(K, M, subset_by_index=[0, n_modes - 1])
    sorted_indices = np.argsort(eigen_vals.real)  # Sort eigenvalues in ascending order

    # Extract sorted eigenvalues and eigenvectors
    eigen_vals = eigen_vals[sorted_indices]
    eigen_vectors = np.real(eigen_vectors[:, sorted_indices])

    # Compute natural frequencies (Hz)
    omega = np.sqrt(eigen_vals.real)

    freq = omega * R2 * np.sqrt(rho * (1 - nu ** 2) / E)
    freqHz = omega / (2 * np.pi)

    # Print the first five natural frequencies
    print("Normalized natural frequencies:\n", freq[0:n_modes:1])
    print("Frequencies (hz):\n", freqHz[0:n_modes:1])

    # Print the first five natural frequencies
    print("Normalized natural frequencies:", freq[:n_modes])

    # Generate a mesh grid for visualization of mode shapes
    xi1 = np.linspace(*rectangular_domain.edges["xi1"], 100)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], 100)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)

    n_modes = 6
    # Create a figure for mode shape visualization
    fig, axes = plt.subplots(1, n_modes, figsize=(15, 5), subplot_kw={'projection': '3d'}, constrained_layout=True)

    # Loop through the first few vibration modes
    for i in range(n_modes):
        mode1, xx = shell.displacement_expansion(eigen_vectors[:, i * 2], x, y)  # Compute mode shape

        mode = reciprocal_base[0] * mode1[0] + reciprocal_base[1] * mode1[1] + reciprocal_base[2] * mode1[2]

        mode = mode / np.max(np.abs(mode)) * 0.01  # Normalize and scale for visualization

        z = shell.mid_surface_geometry(x, y)  # Compute deformed geometry

        ax = axes[i]  # Select subplot
        scmap = plt.cm.ScalarMappable(cmap='jet')  # Define colormap
        ax.plot_surface(z[0, 0] + mode[0], z[1, 0] + mode[1], z[2, 0] + mode[2],
                        facecolors=scmap.to_rgba(mode1[2]),
                        edgecolor='black',
                        linewidth=0.1)  # Plot mode shape

        # Label axes and set the title with frequency information
        ax.set_title(f"Mode {i + 1} - Frequency: {freq[i]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Ensure equal aspect ratio for visualization
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    # Adjust layout and display the plots
    # plt.tight_layout()
    plt.show()
