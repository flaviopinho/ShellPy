import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
from scipy.linalg import eig, eigh

from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy import RectangularMidSurfaceDomain, simply_supported
from shellpy.expansions.polinomial_expansion import GenericPolynomialSeries
from shellpy.materials.functionally_graded_material import FunctionallyGradedMaterial
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.numeric_integration.boole_integral import boole_weights_simple_integral
from shellpy.tensor_derivatives import tensor_derivative
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_

if __name__ == "__main__":
    integral_x = 30
    integral_y = 30
    integral_z = 20

    aRx = 1
    aRy = -1
    ah = 10

    a = 1
    b = 1
    Rx = a / aRx
    Ry = a / aRy

    h = a / ah

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    R_ = sym.Matrix([
        xi1_,  # x
        xi2_,  # y
        1 / (2 * Rx) * (xi1_ - a / 2) ** 2 + 1 / (2 * Ry) * (xi2_ - b / 2) ** 2  # z
    ])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)

    E = 1  # E_1
    nu = 0.3
    rho = 1

    DD = E * h ** 3 / (12 * (1 - nu ** 2))

    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, rho)

    n_modos_1 = 10
    expansion_size = {"u1": (n_modos_1, n_modos_1),
                      "u2": (n_modos_1, n_modos_1),
                      "u3": (n_modos_1, n_modos_1),
                      "v1": (n_modos_1, n_modos_1),
                      "v2": (n_modos_1, n_modos_1),
                      "v3": (0, 0)}

    simply_supported = {"u1": {"xi1": ("S", "S"),
                               "xi2": ("S", "S")},
                        "u2": {"xi1": ("S", "S"),
                               "xi2": ("S", "S")},
                        "u3": {"xi1": ("C", "C"),
                               "xi2": ("C", "C")},
                        "v1": {"xi1": ("S", "S"),
                               "xi2": ("S", "S")},
                        "v2": {"xi1": ("S", "S"),
                               "xi2": ("S", "S")},
                        "v3": {"xi1": ("S", "S"),
                               "xi2": ("S", "S")}}

    #displacement_field = EigenFunctionExpansion(expansion_size, rectangular_domain, simply_supported)
    #displacement_field = EnrichedCosineExpansion(expansion_size, rectangular_domain, simply_supported)
    displacement_field = GenericPolynomialSeries(np.polynomial.Legendre, expansion_size, rectangular_domain, simply_supported)

    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, None)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    T = fosd2_kinetic_energy(shell, integral_x, integral_y, integral_z, boole_weights_simple_integral)
    U2p = fosd2_quadratic_strain_energy(shell, integral_x, integral_y, integral_z, boole_weights_simple_integral)

    # Compute the mass (M) and stiffness (K) matrices
    M = tensor_derivative(tensor_derivative(T, 0), 1)  # Second derivative of kinetic energy (mass matrix)
    K = tensor_derivative(tensor_derivative(U2p, 0), 1)  # Second derivative of strain energy (stiffness matrix)

    # Number of modes to be analyzed
    n_modes = 10

    # Solve the eigenvalue problem for natural frequencies and mode shapes
    eigen_vals, eigen_vectors = eig(K, M)
    sorted_indices = np.argsort(eigen_vals.real)  # Sort eigenvalues in ascending order

    # Extract sorted eigenvalues and eigenvectors
    eigen_vals = eigen_vals[sorted_indices]
    eigen_vectors = np.real(eigen_vectors[:, sorted_indices])

    # Compute natural frequencies (Hz)
    omega = np.sqrt(eigen_vals.real)

    freq = omega * a * b * np.sqrt(rho * h / DD)
    freq = omega * a * b * np.sqrt(rho * h / E)
    freqHz = omega / (2 * np.pi)

    # Print the first five natural frequencies
    print("Normalized natural frequencies:\n", freq[0:n_modes:1])
    # print("Frequencies (Hz):\n", freqHz[0:n_modes:1])

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
        ax.set_title(f"Mode {i + 1} - Frequency: {freq[i]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Ensure equal aspect ratio for visualization
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    # Adjust layout and display the plots
    # plt.tight_layout()
    plt.show()
