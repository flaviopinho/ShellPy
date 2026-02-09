"""

"""
import os
import matplotlib.pyplot as plt
import sympy as sym
import numpy as np

from shellpy.continuationpy.continuation import Continuation
from exemples.nonlinear_static_analysis.residue_jacobian_stability import shell_jacobian, shell_residue, shell_stability
from shellpy.cache_decorator import clear_cache
from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy.fsdt_tensor.fosd_load_energy import fosd_load_energy
from shellpy.fsdt_tensor.fosd_strain_energy import fosd_strain_energy
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.tensor_derivatives import tensor_derivative
from shellpy.shell_loads.shell_conservative_load import ConcentratedForce
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_
import dill



def output_results(shell, xi1, xi2, x, *args):
    u = x[:-1]
    p = x[-1]
    U = shell.displacement_expansion(u, xi1, xi2)
    N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(xi1, xi2)
    U = U[0] * N1 + U[1] * N2 + U[2] * N3

    plot_shell_arc(shell, u)

    return -U[2], p/2, "u_z", "P"


def plot_shell_arc(shell, u):
    """
    Plots the deformed shell geometry.

    Args:
        shell: The Shell object.
        u: Displacement coefficients.
    """
    # Create meshgrid of xi1 and xi2 coordinates for plotting.
    xi1 = np.linspace(*shell.mid_surface_domain.edges["xi1"], 50)
    xi2 = np.linspace(*shell.mid_surface_domain.edges["xi2"], 50)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)

    # Calculate the deformed shape (mode) using the displacement expansion.
    mode1 = shell.displacement_expansion(u, x, y)   # Compute mode shape
    mode1 = mode1[0:3]
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

        data = np.loadtxt("semicylindrical_ansys.txt", delimiter=";")

        x = data[:, 0]
        y = data[:, 1] / 1000
        ax = plt.subplot(1, 2, 1)
        ax.plot(x, y, linestyle='-', color='k')

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

def create_shell():
    integral_x = 6
    integral_y = 6
    integral_z = 4

    R = 1.016
    L = 3.048
    h = 0.03

    density = 1

    E1 = 2.0685E7
    E = 1
    nu = 0.3

    load = ConcentratedForce(0, 0, -1000 / E1, 1, 0)

    rectangular_domain = RectangularMidSurfaceDomain(0, 1, 0, np.pi / 2)

    n_modos = 3
    n_modos1 = 3

    expansion_size = {"u1": (n_modos1, n_modos),
                      "u2": (n_modos1, n_modos),
                      "u3": (n_modos1, n_modos),
                      "v1": (n_modos1, n_modos),
                      "v2": (n_modos1, n_modos),
                      "v3": (n_modos1, n_modos)}

    boundary_conditions_u1 = {"xi1": ("S", "F"),
                              "xi2": ("F", "F")}
    boundary_conditions_u2 = {"xi1": ("S", "F"),
                              "xi2": ("S", "S")}
    boundary_conditions_u3 = {"xi1": ("C", "F"),
                              "xi2": ("FC", "FC")}

    boundary_conditions_v1 = {"xi1": ("S", "F"),
                              "xi2": ("F", "F")}
    boundary_conditions_v2 = {"xi1": ("S", "F"),
                              "xi2": ("S", "S")}
    boundary_conditions_v3 = {"xi1": ("S", "F"),
                              "xi2": ("F", "F")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3,
                           "v1": boundary_conditions_v1,
                           "v2": boundary_conditions_v2,
                           "v3": boundary_conditions_v3}


    #displacement_field = EnrichedCosineExpansion(expansion_size, rectangular_domain, boundary_conditions)
    displacement_field = EigenFunctionExpansion(expansion_size, rectangular_domain, boundary_conditions)
    #displacement_field = GenericPolynomialSeries(np.polynomial.Legendre, expansion_size, rectangular_domain, boundary_conditions)

    R_ = sym.Matrix([xi1_ * L, R * sym.sin(xi2_), R * sym.cos(xi2_)])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, density)
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, load)

    U_ext = fosd_load_energy(shell)

    U2_int, U3_int, U4_int = fosd_strain_energy(shell, integral_x, integral_y, integral_z)

    clear_cache(shell)

    dill.settings['recurse'] = True

    with open("semicylindrical_shell_subjected_to_point_load.dill", "wb") as f:
        dill.dump((shell, U_ext, U2_int, U3_int, U4_int), f)


if __name__ == "__main__":
    filename = "semicylindrical_shell_subjected_to_point_load.dill"
    if not os.path.exists(filename):
        print("Calculating shell")
        create_shell()

    with open(filename, "rb") as f:
        shell, U_ext, U2_int, U3_int, U4_int = dill.load(f)

    # Numero de variaveis
    n = shell.displacement_expansion.number_of_degrees_of_freedom()
    # Numero de parametros
    p = 1

    div = 1
    F_ext = tensor_derivative(U_ext, 0)

    F2_int = tensor_derivative(U2_int, 0)
    F3_int = tensor_derivative(U3_int, 0)
    F4_int = tensor_derivative(U4_int, 0)

    J2_int = tensor_derivative(F2_int, 1)
    J3_int = tensor_derivative(F3_int, 1)
    J4_int = tensor_derivative(F4_int, 1)

    residue = lambda u, *args: shell_residue((F2_int, F3_int, F4_int), F_ext, u, *args)
    jacobian = lambda u, *args: shell_jacobian((J2_int, J3_int, J4_int), F_ext, u, *args)
    stability = shell_stability
    output = lambda u, *args: output_results(shell, 1, 0, u, *args)

    # Limites de interesse das variaveis e parametros
    continuation_boundary = np.zeros((n + p, 2))
    continuation_boundary[:-1, 0] = -100000
    continuation_boundary[:-1, 1] = 100000
    continuation_boundary[-1, 0] = -2
    continuation_boundary[-1, 1] = 10

    # Definindo continuation_model
    continuation_model = {'n': n,
                          'p': 1,
                          'residue': residue,
                          'jacobian': jacobian,
                          'stability_check': stability,
                          'boundary': continuation_boundary,
                          'output_function': output}

    continuation = Continuation(continuation_model)
    continuation.parameters['tol2'] = 1E-9
    continuation.parameters['tol1'] = 1E-9
    continuation.parameters['index1'] = -1
    continuation.parameters['index2'] = 0
    continuation.parameters['cont_max'] = 10000

    continuation.parameters['h_max'] = 100

    # Determinacao de um ponto regular inicial
    u0 = np.zeros(continuation_model['n'] + continuation_model['p'])
    u0[-1] = 0.01
    H0 = continuation.model['residue'](u0)
    J0 = continuation.model['jacobian'](u0)
    t0 = continuation.tangent_vector(J0)
    w0 = 1

    continuation.continue_branch(u0, t0, w0, 'Branch1')

    plt.show()
