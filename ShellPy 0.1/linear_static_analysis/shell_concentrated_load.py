from time import time
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

from expansions.eigen_function_expansion import EigenFunctionExpansion
from expansions.polinomial_expansion import GenericPolynomialSeries
from mid_surface_domain import RectangularMidSurfaceDomain
from tensor_derivatives import tensor_derivative
from koiter_shell_theory.fast_koiter_strain_energy import fast_koiter_quadratic_strain_energy
from koiter_shell_theory.koiter_load_energy import koiter_load_energy
from shell_loads.shell_conservative_load import PressureLoad, ConcentratedForce
from displacement_expansion import simply_supported
from material import LinearElasticMaterial
from shell import Shell
from thickness import ConstantThickness
from midsurface_geometry import MidSurfaceGeometry, xi1_, xi2_

if __name__ == "__main__":
    n_integral_default = 20

    plot_scale = 40

    # Entrada de dados
    # painel esferico
    R = 1
    a = 0.1
    b = 0.1
    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)
    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])

    # placa
    """
    a = 0.1
    b = 0.2
    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)
    R_ = sym.Matrix([xi1_, xi2_, 0])
    """

    # painel cilindrico
    """
    R = 1
    b = 2
    rectangular_domain = RectangularMidSurfaceDomain(-R*0.999, R*0.999, 0, b)
    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_) ** 2)])
    """

    h = 0.001
    density = 7850

    E = 206E9
    nu = 0.3

    load = ConcentratedForce(0, 0, -700, a / 2, b / 2)
    # load = PressureLoad(10)

    expansion_size = {"u1": (6, 6),
                      "u2": (6, 6),
                      "u3": (6, 6)}

    # boundary_conditions = pinned
    # boundary_conditions = fully_clamped
    boundary_conditions = simply_supported

    """
    boundary_conditions_u1 = {"xi1": ("S", "F"), #F: livre (Free)
                              "xi2": ("S", "F")} #S: apoiado (Simply supported
    boundary_conditions_u2 = {"xi1": ("S", "F"), #C: Engastado (clamped)
                              "xi2": ("S", "F")}
    boundary_conditions_u3 = {"xi1": ("C", "F"),
                              "xi2": ("C", "F")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3}
    """

    displacement_field = EigenFunctionExpansion(expansion_size, rectangular_domain, boundary_conditions)
    displacement_field = GenericPolynomialSeries(np.polynomial.Legendre, expansion_size, rectangular_domain,
                                                 boundary_conditions)

    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = LinearElasticMaterial(E, nu, density)
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, load)

    # Analise
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    U1 = koiter_load_energy(shell)

    start = time()
    U2 = fast_koiter_quadratic_strain_energy(shell)
    stop = time()
    print("tempo de cálculo do funcional quadrático em paralelo: ", stop - start)

    F = -tensor_derivative(U1, 0)
    K = tensor_derivative(tensor_derivative(U2, 0), 1)

    U = np.linalg.inv(K) @ F

    # Geração da malha para plotagem
    xi1 = np.linspace(*rectangular_domain.edges["xi1"], 100)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], 100)
    x, y = np.meshgrid(xi1, xi2, indexing='xy')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    mode = shell.displacement_expansion(U, x, y)
    mode_norm = mode / np.max(np.abs(mode)) * h * plot_scale  # Normalizar e escalar
    z = shell.mid_surface_geometry(x, y)

    # Criar um mapa de cores
    cmap = plt.cm.jet  # Escolha um colormap bonito (viridis, plasma, inferno, etc.)
    scmap = plt.cm.ScalarMappable(cmap=cmap)
    scmap.set_array(mode[2])  # Definir valores para a barra de cores

    # Plotar a superfície
    surf = ax.plot_surface(z[0, 0] + mode_norm[0], z[1, 0] + mode_norm[1], z[2, 0] + mode_norm[2],
                           facecolors=cmap((mode[2] - mode[2].min()) / (mode[2].max() - mode[2].min())),
                           edgecolor='k', linewidth=0.2, alpha=0.9)

    # Adicionar uma barra de cores
    cbar = fig.colorbar(scmap, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label("Deslocamento")

    # Configurar os rótulos dos eixos
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Ajustar a plot_scale dos eixos para serem proporcionais
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    max_range = max(x_limits[1] - x_limits[0],
                    y_limits[1] - y_limits[0],
                    z_limits[1] - z_limits[0]) / 2

    mid_x = (x_limits[1] + x_limits[0]) / 2
    mid_y = (y_limits[1] + y_limits[0]) / 2
    mid_z = (z_limits[1] + z_limits[0]) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
