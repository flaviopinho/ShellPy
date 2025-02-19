"""
This example analyzes the nonlinear behavior of a doubly curved shell
(a non shallow spherical panel) under a pressure load.  This shell has been
previously studied by Pinho et al. (DOI: 10.1016/j.engstruct.2021.113674).
This script determines the nonlinear static response.
"""

import sys
import matplotlib.pyplot as plt
import sympy as sym
import numpy as np

from shellpy import simply_supported, pinned
from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy.expansions.polinomial_expansion import GenericPolynomialSeries
from shellpy import RectangularMidSurfaceDomain
from shellpy.koiter_shell_theory import fast_koiter_strain_energy
from shellpy.tensor_derivatives import tensor_derivative
from shellpy.koiter_shell_theory.koiter_load_energy import koiter_load_energy
from shellpy.shell_loads.shell_conservative_load import ConcentratedForce, PressureLoad
from shellpy import LinearElasticMaterial
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_

sys.path.append('../../ContinuationPy/ContinuationPy')
import continuation


def pinho_nonshallow_shell_residue(F_int, F_ext, x, *args):
    u = x[:-1]
    p = x[-1]
    F_int_tot = np.einsum('ij, j->i', F_int[0], u, optimize=True) + \
                np.einsum('ijk, j, k->i', F_int[1], u, u, optimize=True) + \
                np.einsum('ijkl, j, k, l->i', F_int[2], u, u, u, optimize=True)
    return F_int_tot + F_ext * p


def pinho_nonshallow_shell_jacobian(J_int, F_ext, x, *args):
    u = x[:-1]
    p = [-1]
    J_int_tot = J_int[0] + \
                np.einsum('ijk, k->ij', J_int[1], u, optimize=True) + \
                np.einsum('ijkl, k, l->ij', J_int[2], u, u, optimize=True)

    return np.hstack((J_int_tot, F_ext[:, np.newaxis]))


def pinho_nonshallow_shell_stability(u, J, model, *args):
    # Extrai a submatriz Jx
    Jx = J[:model['n'], :model['n']]

    # Determinação dos autovalores de Jx
    eigen_values = np.linalg.eigvals(Jx)

    # Partes reais e imaginárias dos autovalores
    real_part = np.real(eigen_values)
    imaginary_part = np.imag(eigen_values)

    # Estabilidade: maior parte real
    stability = np.max(real_part)

    tipo = None
    if 'tipo' in locals():  # Para verificar se a variável tipo foi definida
        # Análise do tipo de bifurcação
        index_real_positivo = real_part > 0
        index_real_negativo = real_part < 0
        num_real_positivo = np.sum(index_real_positivo)
        num_real_negativo = np.sum(index_real_negativo)

        if num_real_positivo == 2 and np.any(imaginary_part[index_real_positivo] != 0):
            tipo = 'H'  # Hopf
        elif num_real_positivo == 1 and num_real_negativo >= 0:
            tipo = 'SN'  # Ponto de sela
        elif num_real_positivo == 0 and num_real_negativo >= 0:
            tipo = 'PR'  # Ponto regular
        else:
            tipo = 'BC'  # Bifurcação complexa

    return stability, tipo


def pinho_output_results(shell, xi1, xi2, x, *args):
    u = x[:-1]
    p = x[-1]
    U = shell.displacement_expansion(u, xi1, xi2)
    N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(xi1, xi2)
    U = U[0] * N1 + U[1] * N2 + U[2] * N3

    plot_shell(shell, u)

    return -U[2], p, "u_3(pi/2, 0)", "P"


def plot_shell(shell, u):
    """
    Plots the deformed shell geometry.

    Args:
        shell: The Shell object.
        u: Displacement coefficients.
    """
    # Create meshgrid of xi1 and xi2 coordinates for plotting.
    xi1 = np.linspace(*shell.mid_surface_domain.edges["xi1"], 100)
    xi2 = np.linspace(*shell.mid_surface_domain.edges["xi2"], 2)
    x, y = np.meshgrid(xi1, xi2, indexing='xy')

    # Calculate the deformed shape (mode) using the displacement expansion.
    mode = shell.displacement_expansion(u, x, y)  # Compute mode shape

    # Calculate the original (undeformed) mid-surface geometry.
    z = shell.mid_surface_geometry(x, y)  # Compute deformed geometry

    # Create the plot figure and axes.
    fig = plt.figure(1)
    n = len(fig.axes)
    if n == 1:  # If it is the first time the figure is created
        fig.clf()  # Clear figure
        ax = fig.add_subplot(1, 2, 1)  # First subplot (not used)
        ax = fig.add_subplot(1, 2, 2, projection='3d')  # Second subplot (3D plot)
    else:  # If figure already exists
        ax = plt.subplot(1, 2, 2)  # Select the 3D subplot

    ax.cla()  # Clear the axes.

    # Create a colormap for visualization.
    scmap = plt.cm.ScalarMappable(cmap='jet')  # Define colormap

    # Plot the deformed shell surface. The displacement is scaled by a factor of 5 for visualization.
    ax.plot_surface(z[0, 0] + mode[0], z[1, 0] + mode[1] , z[2, 0] + mode[2],
                    facecolors=scmap.to_rgba(mode[2]),  # Color based on transverse displacement
                    edgecolor='black',  # Black edges
                    linewidth=0.1)  # Edge linewidth

    # Label the axes.
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Ensure equal aspect ratio for proper visualization of the shell shape.
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    plt.pause(0.01)  # Pause briefly to allow the plot to update. This is important for animations.


if __name__ == "__main__":
    R = 1
    b = 0.1
    alpha = 35 / 2 * (np.pi / 180)
    alpha1 = -alpha
    alpha2 = np.pi + alpha
    h = 0.001

    In = b * h ** 3 / 12

    density = 1

    E = 1
    nu = 0.3

    load = ConcentratedForce(0, 0, 1 / (R ** 2 / (E * In)), np.pi / 2, 0)

    rectangular_domain = RectangularMidSurfaceDomain(alpha1, alpha2, b / 2, -b / 2)

    expansion_size = {"u1": (15, 4),
                      "u2": (2, 2),
                      "u3": (20, 4)}

    boundary_conditions_u1 = {"xi1": ("S", "S"),
                              "xi2": ("F", "F")}
    boundary_conditions_u2 = {"xi1": ("S", "S"),
                              "xi2": ("F", "F")}
    boundary_conditions_u3 = {"xi1": ("C", "C"),
                              "xi2": ("F", "F")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3}

    displacement_field = EnrichedCosineExpansion(expansion_size, rectangular_domain, boundary_conditions)

    R_ = sym.Matrix([R * sym.cos(xi1_), xi2_, R * sym.sin(xi1_)])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = LinearElasticMaterial(E, nu, density)
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, load)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    U_ext = koiter_load_energy(shell)

    U2_int, U3_int, U4_int = fast_koiter_strain_energy(shell)

    # Numero de variaveis
    n = displacement_field.number_of_degrees_of_freedom()
    # Numero de parametros
    p = 1

    div = E * h ** 2
    F_ext = tensor_derivative(U_ext, 0) / div

    F2_int = tensor_derivative(U2_int, 0) * h / div
    F3_int = tensor_derivative(U3_int, 0) * h ** 2 / div
    F4_int = tensor_derivative(U4_int, 0) * h ** 3 / div

    J2_int = tensor_derivative(F2_int, 1)
    J3_int = tensor_derivative(F3_int, 1)
    J4_int = tensor_derivative(F4_int, 1)

    residue = lambda u, *args: pinho_nonshallow_shell_residue((F2_int, F3_int, F4_int), F_ext, u, *args)
    jacobian = lambda u, *args: pinho_nonshallow_shell_jacobian((J2_int, J3_int, J4_int), F_ext, u, *args)
    stability = pinho_nonshallow_shell_stability
    output = lambda u, *args: pinho_output_results(shell, np.pi / 2, 0, u, *args)

    # Limites de interesse das variaveis e parametros
    continuation_boundary = np.zeros((n + p, 2))
    continuation_boundary[:-1, 0] = -100000
    continuation_boundary[:-1, 1] = 100000
    continuation_boundary[-1, 0] = -0.2
    continuation_boundary[-1, 1] = 20

    # Definindo continuation_model
    continuation_model = {'n': n_dof,
                          'p': 1,
                          'residue': residue,
                          'jacobian': jacobian,
                          'stability_check': stability,
                          'boundary': continuation_boundary,
                          'output_function': output}

    continuation = continuation.Continuation(continuation_model)
    continuation.parameters['tol2'] = 1E-9
    continuation.parameters['tol1'] = 1E-5
    continuation.parameters['index1'] = -1
    continuation.parameters['index2'] = 0

    continuation.parameters['h_max'] = 1

    # Determinacao de um ponto regular inicial
    u0 = np.zeros(continuation_model['n'] + continuation_model['p'])
    u0[-1] = 0.1
    H0 = continuation.model['residue'](u0)
    J0 = continuation.model['jacobian'](u0)
    t0 = continuation.tangent_vector(J0)
    w0 = 1

    continuation.continue_branch(u0, t0, w0, 'Branch1')

    plt.show()
