"""
This example analyzes the nonlinear behavior of an arc
 under a concentrated force unsing Koiter-Sanders theory. This arc has been
previously studied (https://doi.org/10.1016/j.ijnonlinmec.2005.06.009).
This script determines the nonlinear static response.
"""

import sys
import matplotlib.pyplot as plt
import sympy as sym
import numpy as np

import shellpy
from nonlinear_static_analysis.residue_jacobian_stability import shell_jacobian, shell_residue, shell_stability
from shellpy import simply_supported, pinned
from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy.expansions.polinomial_expansion import GenericPolynomialSeries
from shellpy import RectangularMidSurfaceDomain
from shellpy.koiter_shell_theory import fast_koiter_strain_energy
from shellpy.koiter_shell_theory.koiter_strain_energy_large import \
    koiter_strain_energy_large_rotations
from shellpy.tensor_derivatives import tensor_derivative
from shellpy.koiter_shell_theory.koiter_load_energy import koiter_load_energy
from shellpy.shell_loads.shell_conservative_load import ConcentratedForce, PressureLoad
from shellpy import LinearElasticMaterial
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_
import shellpy.integral_booles_rule

sys.path.append('../../ContinuationPy/ContinuationPy')
import continuation


def arc_output_results(shell, xi1, xi2, x, *args):
    u = x[:-1]
    p = x[-1]
    h = shell.thickness(xi1, xi2)
    U = shell.displacement_expansion(u, xi1, xi2) * h
    N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(xi1, xi2)
    U = U[0] * N1 + U[1] * N2 + U[2] * N3

    plot_shell_arc(shell, u)

    return -U[2], p, "u_3(pi/2, 0)", "P"


def plot_shell_arc(shell, u):
    """
    Plots the deformed shell geometry.

    Args:
        shell: The Shell object.
        u: Displacement coefficients.
    """
    # Create meshgrid of xi1 and xi2 coordinates for plotting.
    xi1 = np.linspace(*shell.mid_surface_domain.edges["xi1"], 100)
    xi2 = np.linspace(*shell.mid_surface_domain.edges["xi2"], 2)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')
    h = shell.thickness(xi1, xi2)

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)

    # Calculate the deformed shape (mode) using the displacement expansion.
    mode1 = shell.displacement_expansion(u, x, y) * h  # Compute mode shape
    mode = reciprocal_base[0] * mode1[0] + reciprocal_base[1] * mode1[1] + reciprocal_base[2] * mode1[2]

    # Calculate the original (undeformed) mid-surface geometry.
    z = shell.mid_surface_geometry(x, y)  # Compute deformed geometry

    # Create the plot figure and axes.
    fig = plt.figure(1)
    n = len(fig.axes)
    if n == 1:  # If it is the first time the figure is created
        fig.clf()  # Clear figure
        ax = fig.add_subplot(1, 2, 1)  # First subplot (not used)
        ax = fig.add_subplot(1, 2, 2, projection='3d')  # Second subplot (3D plot)

        data = np.loadtxt("arc_results.txt", delimiter=",")

        x = data[:, 0]
        y = data[:, 1]
        ax = plt.subplot(1, 2, 1)
        ax.plot(x, y, linestyle='-', color='r')

        ax = plt.subplot(1, 2, 2)
    else:  # If figure already exists
        ax = plt.subplot(1, 2, 2)  # Select the 3D subplot

    ax.cla()  # Clear the axes.

    # Create a colormap for visualization.
    scmap = plt.cm.ScalarMappable(cmap='jet')  # Define colormap

    # Plot the deformed shell surface. The displacement is scaled by a factor of 5 for visualization.
    ax.plot_surface(z[0, 0] + mode[0], z[1, 0] + mode[1], z[2, 0] + mode[2],
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
    nu = 0

    n_int_x = 40
    n_int_y = 1

    load = ConcentratedForce(0, 0, -1 / (R ** 2 / (E * In)), np.pi / 2, 0)

    rectangular_domain = RectangularMidSurfaceDomain(alpha1, alpha2, -b / 2, b / 2)

    expansion_size = {"u1": (20, 1),
                      "u2": (0, 0),
                      "u3": (20, 1)}

    boundary_conditions_u1 = {"xi1": ("S", "S"),
                              "xi2": ("O", "O")}
    boundary_conditions_u2 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_u3 = {"xi1": ("S", "C"),
                              "xi2": ("O", "O")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3}

    displacement_field = EnrichedCosineExpansion(expansion_size, rectangular_domain, boundary_conditions)

    R_ = sym.Matrix([-R * sym.cos(xi1_), xi2_, R * sym.sin(xi1_)])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = LinearElasticMaterial(E, nu, density)
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, load)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    U_ext = koiter_load_energy(shell)

    U2_int, U3_int, U4_int = koiter_strain_energy_large_rotations(shell, n_int_x, n_int_y)

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

    residue = lambda u, *args: shell_residue((F2_int, F3_int, F4_int), F_ext, u, *args)
    jacobian = lambda u, *args: shell_jacobian((J2_int, J3_int, J4_int), F_ext, u, *args)
    stability = shell_stability
    output = lambda u, *args: arc_output_results(shell, np.pi / 2, 0, u, *args)

    # Limites de interesse das variaveis e parametros
    continuation_boundary = np.zeros((n + p, 2))
    continuation_boundary[:-1, 0] = -100000
    continuation_boundary[:-1, 1] = 100000
    continuation_boundary[-1, 0] = -0.2
    continuation_boundary[-1, 1] = 50

    # Definindo continuation_model
    continuation_model = {'n': n_dof,
                          'p': 1,
                          'residue': residue,
                          'jacobian': jacobian,
                          'stability_check': stability,
                          'boundary': continuation_boundary,
                          'output_function': output}

    continuation = continuation.Continuation(continuation_model)
    continuation.parameters['tol2'] = 1E-6
    continuation.parameters['tol1'] = 1E-6
    continuation.parameters['index1'] = -1
    continuation.parameters['index2'] = 0
    continuation.parameters['cont_max'] = 10000

    continuation.parameters['h_max'] = 100

    # Determinacao de um ponto regular inicial
    u0 = np.zeros(continuation_model['n'] + continuation_model['p'])
    u0[-1] = 0.1
    H0 = continuation.model['residue'](u0)
    J0 = continuation.model['jacobian'](u0)
    t0 = continuation.tangent_vector(J0)
    w0 = 1

    continuation.continue_branch(u0, t0, w0, 'Branch1')

    plt.show()
