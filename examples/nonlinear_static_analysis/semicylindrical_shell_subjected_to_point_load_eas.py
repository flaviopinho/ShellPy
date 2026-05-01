"""

"""
import matplotlib.pyplot as plt
import sympy as sym
import numpy as np

from shellpy.continuationpy.continuation import Continuation
from shellpy.expansions import EnrichedCosineExpansion
from shellpy.fsdt6 import load_energy
from shellpy.fsdt7_eas.EAS_expansion import EasExpansion
from shellpy.fsdt7_eas.internal_force_vector import internal_force_vector
from shellpy.fsdt7_eas.tangent_stiffness_matrix import tangent_stiffness_matrix
from shellpy.utils.residue_jacobian_stability import shell_stability
from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.tensor_derivatives import tensor_derivative
from shellpy.shell_loads.shell_conservative_load import ConcentratedForce
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_


def residuo1(u, shell, eas_field, F_ext, nx, ny, nz):
    return internal_force_vector(u[0:-1], shell, eas_field, nx, ny, nz) + F_ext * u[-1]


def jacobian1(u, shell, eas_field, F_ext, nx, ny, nz):
    J_int = tangent_stiffness_matrix(u[0:-1], shell, eas_field, nx, ny, nz)
    return np.hstack((J_int, F_ext[:, np.newaxis]))


def output_results(shell, xi1, xi2, x, *args):
    u = x[:-1]
    p = x[-1]
    U = shell.displacement_expansion(u, xi1, xi2)
    N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(xi1, xi2)
    U = U[0] * N1 + U[1] * N2 + U[2] * N3

    plot_shell_arc(shell, u)

    return -U[2], p, "u_z", "P"


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
    mode1 = shell.displacement_expansion(u, x, y)  # Compute mode shape
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


if __name__ == "__main__":
    integral_x = 5
    integral_y = 5
    integral_z = 4

    R = 1.016
    L = 3.048
    h = 0.03

    density = 1

    E1 = 2.0685E7
    E = 1
    nu = 0.3

    load = ConcentratedForce(0, 0, -500 / E1, 1, 0)

    rectangular_domain = RectangularMidSurfaceDomain(0, 1, 0, np.pi / 2)

    n_modos = 6
    n_modos1 = 6

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
    boundary_conditions_v3 = {"xi1": ("F", "F"),
                              "xi2": ("F", "F")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3,
                           "v1": boundary_conditions_v1,
                           "v2": boundary_conditions_v2,
                           "v3": boundary_conditions_v3}

    # displacement_field = EnrichedCosineExpansion(expansion_size, rectangular_domain, boundary_conditions)
    displacement_field = EigenFunctionExpansion(expansion_size, rectangular_domain, boundary_conditions)
    # displacement_field = GenericPolynomialSeries(np.polynomial.Legendre, expansion_size, rectangular_domain, boundary_conditions)

    eas_field = EasExpansion({"eas": (n_modos1, n_modos)}, rectangular_domain,
                             {"eas": {"xi1": ("F", "F"), "xi2": ("F", "F")}})

    R_ = sym.Matrix([xi1_ * L, R * sym.sin(xi2_), R * sym.cos(xi2_)])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, density)
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, load)

    U_ext = load_energy(shell)
    F_ext = tensor_derivative(U_ext, 0)

    # Number of degrees of freedom
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Number of variables (degrees of freedom)
    n = displacement_field.number_of_degrees_of_freedom()
    # Number of parameters (external loading factor)
    p = 1

    # Define functions for residual, Jacobian, stability, and output
    residue = lambda u, *args: residuo1(u, shell, eas_field, F_ext, integral_x, integral_y, integral_z)
    jacobian = lambda u, *args: jacobian1(u, shell, eas_field, F_ext, integral_x, integral_y, integral_z)
    stability = shell_stability
    output = lambda u, *args: output_results(shell, 1, 0, u, *args)

    # Limites de interesse das variaveis e parametros
    continuation_boundary = np.zeros((n + p, 2))
    continuation_boundary[:-1, 0] = -100000
    continuation_boundary[:-1, 1] = 100000
    continuation_boundary[-1, 0] = -0.1
    continuation_boundary[-1, 1] = 1.6

    # Definindo continuation_model
    continuation_model = {'n': n,
                          'p': 1,
                          'residue': residue,
                          'jacobian': jacobian,
                          'stability_check': stability,
                          'boundary': continuation_boundary,
                          'output_function': output}

    continuation = Continuation(continuation_model)
    continuation.parameters['tol2'] = 1E-7
    continuation.parameters['tol1'] = 1E-7
    continuation.parameters['index1'] = -1
    continuation.parameters['index2'] = 0
    continuation.parameters['cont_max'] = 10000

    continuation.parameters['h_max'] = 10000
    continuation.parameters['h0'] = 0.01
    continuation.parameters['jacobian_correction'] = True
    continuation.parameters['solver_pinv'] = True

    # Determinacao de um ponto regular inicial
    u0 = np.zeros(continuation_model['n'] + continuation_model['p'])
    u0[-1] = 0.01
    H0 = continuation.model['residue'](u0)
    J0 = continuation.model['jacobian'](u0)
    t0 = continuation.tangent_vector(J0)
    w0 = 1

    continuation.continue_branch(u0, t0, w0, 'Branch1')

    plt.show()
