import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
from scipy.linalg import eig

from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy.fosd_theory2.fosd2_kinetic_energy import fosd2_kinetic_energy
from shellpy.fosd_theory2.fosd2_strain_energy import fosd2_quadratic_strain_energy
from shellpy.materials.laminate_orthotropic_material import Lamina, LaminateOrthotropicMaterial
from shellpy.tensor_derivatives import tensor_derivative
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_

if __name__ == "__main__":
    integral_x = 10
    integral_y = 10
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

    material = LaminateOrthotropicMaterial([lamina45m, lamina45p, lamina45p, lamina45m, lamina45m, lamina45p, lamina45p, lamina45m], thickness)
    #material = IsotropicHomogeneousLinearElasticMaterial(2E11, 0.3, 7850)

    n_modos = 10
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

    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, None)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    T = fosd2_kinetic_energy(shell, integral_x, integral_y, integral_z)
    U2p = fosd2_quadratic_strain_energy(shell, integral_x, integral_y, integral_z)

    #T = fosd_kinetic_energy(shell, integral_x, integral_y, integral_z)
    #U2p = fosd_quadratic_strain_energy(shell, integral_x, integral_y, integral_z)

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

    freq = omega /(2*np.pi)

    # Number of modes to be analyzed
    n_modes = 5

    # Print the first five natural frequencies
    print("Normalized natural frequencies:", freq[:n_modes])

    # Generate a mesh grid for visualization of mode shapes
    xi1 = np.linspace(*rectangular_domain.edges["xi1"], 100)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], 100)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)

    # Create a figure for mode shape visualization
    fig, axes = plt.subplots(1, n_modes, figsize=(15, 5), subplot_kw={'projection': '3d'}, constrained_layout=True)

    # Loop through the first few vibration modes
    for i in range(n_modes):
        mode1, xx = shell.displacement_expansion(eigen_vectors[:, i], x, y)  # Compute mode shape

        mode = reciprocal_base[0] * mode1[0] + reciprocal_base[1] * mode1[1] + reciprocal_base[2] * mode1[2]

        mode = mode / np.max(np.abs(mode)) * h * 10  # Normalize and scale for visualization

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
