"""
This example analyzes the nonlinear behavior of a doubly curved shell
(a non shallow spherical panel) under a pressure load.  This shell has been
previously studied by Pinho et al. (DOI: 10.1016/j.engstruct.2021.113674).
This script determines the nonlinear static response.
"""

import matplotlib.pyplot as plt
import sympy as sym
import numpy as np

from continuationpy.continuation import Continuation
from exemples.nonlinear_static_analysis.residue_jacobian_stability import shell_stability, shell_jacobian, shell_residue
from shellpy import pinned
from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy import RectangularMidSurfaceDomain

from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.sanders_koiter import koiter_load_energy, fast_koiter_strain_energy
from shellpy.tensor_derivatives import tensor_derivative

from shellpy.shell_loads.shell_conservative_load import PressureLoad
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_

def non_shallow_sphere_panel_output_results(shell, xi1, xi2, x, *args):
    u = x[:-1]
    p = x[-1]
    U = shell.displacement_expansion(u, xi1, xi2)
    N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(xi1, xi2)
    U = U[0] * N1 + U[1] * N2 + U[2] * N3

    # Plot the deformed shell.
    plot_shell(shell, u)

    return p, U[2], "p (1E7)", "u_3(a/4,b/4)/h"


def plot_shell(shell, u):
    """
    Plots the deformed shell geometry.

    Args:
        shell: The Shell object.
        u: Displacement coefficients.
    """
    # Create meshgrid of xi1 and xi2 coordinates for plotting.
    xi1 = np.linspace(*shell.mid_surface_domain.edges["xi1"], 30)
    xi2 = np.linspace(*shell.mid_surface_domain.edges["xi2"], 30)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)

    # Calculate the deformed shape (mode) using the displacement expansion.
    mode1 = shell.displacement_expansion(u, x, y) * h  # Compute mode shape
    mode = reciprocal_base[0] * mode1[0] + reciprocal_base[1] * mode1[1] + reciprocal_base[2] * mode1[2]

    # Calculate the original (undeformed) mid-surface geometry.
    z = shell.mid_surface_geometry(x, y)  # Compute deformed geometry

    # Create the plot figure and axes.
    fig = plt.figure(1)
    n = len(fig.axes)
    if n == 1: # If it is the first time the figure is created
        fig.clf() # Clear figure
        ax = fig.add_subplot(1, 2, 1) # First subplot (not used)
        ax = fig.add_subplot(1, 2, 2, projection='3d') # Second subplot (3D plot)

        data = np.loadtxt("non_shallow_sphere_abaqus.txt", delimiter=None, skiprows=1)
        xx = data[0:440, 0] / 100000
        yy = data[0:440, 2] / 0.001

        ax = plt.subplot(1, 2, 1)
        ax.plot(xx, yy, linestyle='-', color='k', label='Abaqus')

        ax.legend()

        ax = plt.subplot(1, 2, 2)
    else: # If figure already exists
        ax = plt.subplot(1, 2, 2) # Select the 3D subplot

    ax.cla()  # Clear the axes.

    # Create a colormap for visualization.
    scmap = plt.cm.ScalarMappable(cmap='jet')  # Define colormap

    # Plot the deformed shell surface. The displacement is scaled by a factor of 5 for visualization.
    ax.plot_surface(z[0, 0] + mode[0], z[1, 0] + mode[1], z[2, 0] + mode[2],
                    facecolors=scmap.to_rgba(mode[2]), # Color based on transverse displacement
                    edgecolor='black', # Black edges
                    linewidth=0.1)  # Edge linewidth

    # Label the axes.
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Ensure equal aspect ratio for proper visualization of the shell shape.
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    plt.pause(0.01)  # Pause briefly to allow the plot to update. This is important for animations.


if __name__ == "__main__":

    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.001
    density = 7850

    E = 1
    E1 = 206E9
    nu = 0.3

    load = PressureLoad(-1E7 / E1)

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    expansion_size = {"u1": (6, 6),
                      "u2": (6, 6),
                      "u3": (6, 6)}

    boundary_conditions = pinned

    mapping = []
    modes_xi1 = [2, 4, 6, 8, 10, 12, 14, 16]
    modes_xi2 = [1, 3, 5, 7, 9, 11, 13, 15]
    for i in range(expansion_size["u1"][0]):
        for j in range(expansion_size["u1"][1]):
            mapping.append(("u1", modes_xi1[i], modes_xi2[j]))

    modes_xi1 = [1, 3, 5, 7, 9, 11, 13, 15]
    modes_xi2 = [2, 4, 6, 8, 10, 12, 14, 16]
    for i in range(expansion_size["u2"][0]):
        for j in range(expansion_size["u2"][1]):
            mapping.append(("u2", modes_xi1[i], modes_xi2[j]))

    modes_xi1 = [1, 3, 5, 7, 9, 11, 13, 15]
    modes_xi2 = [1, 3, 5, 7, 9, 11, 13, 15]
    for i in range(expansion_size["u3"][0]):
        for j in range(expansion_size["u3"][1]):
            mapping.append(("u3", modes_xi1[i], modes_xi2[j]))

    displacement_field = EigenFunctionExpansion(expansion_size, rectangular_domain, boundary_conditions, mapping)
    # displacement_field = GenericPolynomialSeries(np.polynomial.Legendre, expansion_size, mid_surface_domain, boundary_conditions)

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, density)
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, load)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    U_ext = koiter_load_energy(shell)

    U2_int, U3_int, U4_int = fast_koiter_strain_energy(shell)
    #U2_int, U3_int, U4_int = koiter_strain_energy_large_rotations(shell)

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
    output = lambda u, *args: non_shallow_sphere_panel_output_results(shell, a / 4, b / 4, u, *args)

    # Limites de interesse das variaveis e parametros
    continuation_boundary = np.zeros((n + p, 2))
    continuation_boundary[:-1, 0] = -1000
    continuation_boundary[:-1, 1] = 1000
    continuation_boundary[-1, 0] = -2
    continuation_boundary[-1, 1] = 2

    # Definindo continuation_model
    continuation_model = {'n': n_dof,
                          'p': 1,
                          'residue': residue,
                          'jacobian': jacobian,
                          'stability_check': stability,
                          'boundary': continuation_boundary,
                          'output_function': output}

    continuation = Continuation(continuation_model)
    continuation.parameters['tol2'] = 0
    continuation.parameters['tol1'] = 1E-5
    continuation.parameters['index1'] = -1
    continuation.parameters['index2'] = mapping.index(('u3', 1, 1))

    continuation.parameters['h_max'] = 0.1

    # Determinacao de um ponto regular inicial
    u0 = np.zeros(continuation_model['n'] + continuation_model['p'])
    u0[-1] = 0.01
    H0 = continuation.model['residue'](u0)
    J0 = continuation.model['jacobian'](u0)
    t0 = continuation.tangent_vector(J0)
    w0 = 1

    continuation.continue_branch(u0, t0, w0, 'Branch1')

    plt.show()
