import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy import xi1_, xi2_, MidSurfaceGeometry
from shellpy import Shell
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.sanders_koiter import fast_koiter_kinetic_energy, fast_koiter_quadratic_strain_energy
from shellpy.tensor_derivatives import tensor_derivative
from shellpy import ConstantThickness


if __name__ == "__main__":

    # --------------------------------------------------------------
    # Geometry parameters (corrugated shell)
    # --------------------------------------------------------------
    n = 20       # circumferential waves
    p = 20        # radial exponent
    L = 1.0      # radial length
    R_in = 0.3   # inner radius
    H = 0.3      # corrugation amplitude

    h = L / 100
    density = 1

    # --------------------------------------------------------------
    # Material
    # --------------------------------------------------------------
    E = 1
    nu = 0.3

    # --------------------------------------------------------------
    # Integration points
    # --------------------------------------------------------------
    n_int_x = 20
    n_int_y = n * 8
    n_int_z = 4

    # --------------------------------------------------------------
    # Domain (radial + angular)
    # --------------------------------------------------------------
    rectangular_domain = RectangularMidSurfaceDomain(R_in, R_in + L, 0, 2 * np.pi)

    # --------------------------------------------------------------
    # Expansion size (reduzido para estabilidade inicial)
    # --------------------------------------------------------------
    expansion_size = {
        "u1": (20, n*4),
        "u2": (20, n*4),
        "u3": (20, n*4)
    }

    # --------------------------------------------------------------
    # Boundary conditions
    # xi1: radial direction
    # xi2: circumferential (PERIÓDICO!)
    # --------------------------------------------------------------
    boundary_conditions_u1 = {
        "xi1": ("S", "F"),
        "xi2": ("R", "R")
    }

    boundary_conditions_u2 = {
        "xi1": ("S", "F"),
        "xi2": ("R", "R")
    }

    boundary_conditions_u3 = {
        "xi1": ("C", "F"),
        "xi2": ("R", "R")
    }

    boundary_conditions = {
        "u1": boundary_conditions_u1,
        "u2": boundary_conditions_u2,
        "u3": boundary_conditions_u3
    }

    # --------------------------------------------------------------
    # Displacement field
    # --------------------------------------------------------------
    displacement_field = EnrichedCosineExpansion(
        expansion_size,
        rectangular_domain,
        boundary_conditions
    )

    # --------------------------------------------------------------
    # Mid-surface geometry
    # --------------------------------------------------------------
    R_ = sym.Matrix([
        xi1_ * sym.cos(xi2_),
        xi1_ * sym.sin(xi2_),
        H * ((xi1_ - R_in) / L)**p * sym.cos(n * xi2_)
    ])

    mid_surface_geometry = MidSurfaceGeometry(R_)

    # --------------------------------------------------------------
    # Shell object
    # --------------------------------------------------------------
    thickness = ConstantThickness(h)
    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, density)

    shell = Shell(
        mid_surface_geometry,
        thickness,
        rectangular_domain,
        material,
        displacement_field,
        None
    )

    # --------------------------------------------------------------
    # DOFs
    # --------------------------------------------------------------
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # --------------------------------------------------------------
    # Energy
    # --------------------------------------------------------------
    T = fast_koiter_kinetic_energy(shell, n_int_x, n_int_y, n_int_z)
    U2p = fast_koiter_quadratic_strain_energy(shell, n_int_x, n_int_y, n_int_z)

    # --------------------------------------------------------------
    # Matrices
    # --------------------------------------------------------------
    M = tensor_derivative(tensor_derivative(T, 0), 1)
    K = tensor_derivative(tensor_derivative(U2p, 0), 1)

    # --------------------------------------------------------------
    # Eigenvalue problem
    # --------------------------------------------------------------
    eigen_vals, eigen_vectors = eig(K, M, right=True)

    sorted_indices = np.argsort(eigen_vals.real)
    eigen_vals = eigen_vals[sorted_indices]
    eigen_vectors = np.real(eigen_vectors[:, sorted_indices])

    omega = np.sqrt(eigen_vals.real)
    freq = omega / (2 * np.pi)

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------
    n_modes = 5
    print("Natural frequencies (Hz):")
    print(freq[:n_modes])

    # --------------------------------------------------------------
    # Visualization grid
    # --------------------------------------------------------------
    xi1 = np.linspace(*rectangular_domain.edges["xi1"], n_int_x * 4)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], n_int_y * 4)

    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------
    fig, axes = plt.subplots(
        2, n_modes,
        figsize=(20, 6),
        subplot_kw={'projection': '3d'},
        constrained_layout=True
    )

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)

    for j in range(2):
        for i in range(n_modes):

            m = j * n_modes + i

            mode1 = shell.displacement_expansion(eigen_vectors[:, m], x, y)

            mode = (
                reciprocal_base[0] * mode1[0] +
                reciprocal_base[1] * mode1[1] +
                reciprocal_base[2] * mode1[2]
            )

            mode = mode / np.max(np.abs(mode)) * 0.02

            z = shell.mid_surface_geometry(x, y)

            ax = axes[j, i]

            scmap = plt.cm.ScalarMappable(cmap='viridis')

            ax.plot_surface(
                z[0, 0] + mode[0],
                z[1, 0] + mode[1],
                z[2, 0] + mode[2],
                facecolors=scmap.to_rgba(mode1[2]),
                edgecolor='none'
            )

            ax.set_title(f"Mode {m+1} - {freq[m]:.2f} Hz")

            ax.set_box_aspect([1, 1, 0.5])

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

    plt.show()

    plt.figure(figsize=(8, 5))

    plt.hist(freq, bins=30)

    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Quantidade de modos")
    plt.title("Histograma das frequências naturais")

    plt.grid(True)

    plt.show()

    # Adicione isso após o cálculo das frequências
    plt.figure(figsize=(8, 5))
    plt.step(freq, np.arange(1, len(freq) + 1), where='post')
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("N(f) - Número acumulado de modos")
    plt.title("Função de Contagem de Modos (Efeito da Densidade Modal)")
    plt.grid(True)
    plt.show()