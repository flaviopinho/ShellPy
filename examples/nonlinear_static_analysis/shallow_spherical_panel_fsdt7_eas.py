"""
This example analyzes the nonlinear behavior of a doubly curved shell
(a spherical panel) under a concentrated central load.  This shell has been
previously studied by Kobayashi and Leissa (https://doi.org/10.1016/0020-7462(94)00030-E),
Amabili (https://doi.org/10.1016/j.ijnonlinmec.2004.08.007),
and Pinho et al. (DOI: 10.1016/j.engstruct.2021.113674).  This script determines
the nonlinear static response.
"""

import matplotlib.pyplot as plt
import sympy as sym
import numpy as np

from shellpy.continuationpy.continuation import Continuation
from shellpy.fsdt6 import load_energy
from shellpy.fsdt7_eas.EAS_expansion import EasExpansion
from shellpy.fsdt7_eas.internal_force_vector import internal_force_vector
from shellpy.fsdt7_eas.tangent_stiffness_matrix import tangent_stiffness_matrix
from shellpy.utils.residue_jacobian_stability import shell_stability
# Import custom modules related to shell analysis
from shellpy.displacement_expansion import simply_supported, SSSS_fsdt6  # Defines simply supported boundary conditions
from shellpy.expansions.eigen_function_expansion import \
    EigenFunctionExpansion  # Eigenfunction expansion for displacements
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.mid_surface_domain import RectangularMidSurfaceDomain  # Defines the geometry of the mid-surface
from shellpy.tensor_derivatives import tensor_derivative  # For calculating tensor derivatives (likely for strain)
from shellpy.shell_loads.shell_conservative_load import ConcentratedForce  # Defines a concentrated force
from shellpy.shell import Shell  # Represents the shell structure
from shellpy.thickness import ConstantThickness  # Defines constant thickness
from shellpy.midsurface_geometry import MidSurfaceGeometry, xi1_, \
    xi2_  # Defines the mid-surface geometry and coordinate system


def residuo1(u, shell, eas_field, F_ext, nx, ny, nz):
    return internal_force_vector(u[0:-1], shell, eas_field, nx, ny, nz) + F_ext * u[-1]


def jacobian1(u, shell, eas_field, F_ext, nx, ny, nz):
    J_int = tangent_stiffness_matrix(u[0:-1], shell, eas_field, nx, ny, nz)
    return np.hstack((J_int, F_ext[:, np.newaxis]))


def shallow_spherical_panel_output_results(shell, xi1, xi2, x, *args):
    """
    Processes and outputs results, including plotting the deformed shell.

    Args:
        shell: The Shell object.
        xi1: Array of xi1 coordinates for evaluation.
        xi2: Array of xi2 coordinates for evaluation.
        x: Vector of unknowns (displacement coefficients and load parameter).
        *args: Additional arguments (not used here).

    Returns:
        p: The load parameter.
        U[2]: The transverse displacement at the center (a/4, b/4).
        "F": String indicating the force type (likely "F" for force).
        "u_3(a/4,b/4)": String representing the transverse displacement at the center.
    """
    u = x[:-1]  # Extract displacement coefficients.
    p = x[-1]  # Extract the load parameter.

    # Calculate the displacement field (U) using the displacement expansion.
    U = shell.displacement_expansion(u, xi1, xi2)

    # Get the reciprocal base vectors (N1, N2, N3) from the mid-surface geometry.
    N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(xi1, xi2)

    # Combine the displacement components with the reciprocal base vectors to get the
    # displacement vector U.
    U = U[0] * N1 + U[1] * N2 + U[2] * N3

    # Plot the deformed shell.
    plot_shell(shell, u)

    return p, U[2] / shell.thickness(), "F", "u_3(a/4,b/4)/h"  # Return load parameter, center displacement, and labels.


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
    if n == 1:  # If it is the first time the figure is created
        fig.clf()  # Clear figure
        ax = fig.add_subplot(1, 2, 1)  # First subplot (not used)
        ax = fig.add_subplot(1, 2, 2, projection='3d')  # Second subplot (3D plot)

        data = np.loadtxt("shallow_sphere_abaqus.txt", delimiter=";")

        xx = -data[:, 0]
        yy = data[:, 1] / 0.001
        ax = plt.subplot(1, 2, 1)
        ax.plot(xx, yy, linestyle='-', color='k', label='Abaqus')

        ax.legend()

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
    # Define shell parameters
    R = 1  # Radius of curvature
    a = 0.1  # Length in the x-direction
    b = 0.1  # Length in the y-direction
    h = 0.001  # Shell thickness
    density = 7850  # Material density

    # Material properties
    E = 206E9  # Young's modulus
    nu = 0.3  # Poisson's ratio

    nx, ny, nz = 8, 8, 4

    # Define external concentrated force at the center of the shell
    load = ConcentratedForce(0, 0, -1, a / 2, b / 2)

    # Define the rectangular domain representing the mid-surface of the shell
    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    modes = 6
    # Define the expansion size for displacement fields
    expansion_size = {"u1": (modes, modes),  # Number of modes for in-plane displacement u1
                      "u2": (modes, modes),  # Number of modes for in-plane displacement u2
                      "u3": (modes, modes),  # Number of modes for transverse displacement u3
                      "v1": (modes, modes),  # Number of modes for in-plane displacement v1
                      "v2": (modes, modes),  # Number of modes for in-plane displacement v2
                      "v3": (modes, modes)}  # Number of modes for transverse displacement v3

    # Set boundary conditions (simply supported case)
    boundary_conditions = SSSS_fsdt6

    # Define the displacement field expansion using eigenfunctions
    displacement_field = EigenFunctionExpansion(expansion_size, rectangular_domain, boundary_conditions)

    eas_field = EasExpansion({"eas": (modes, modes)}, rectangular_domain,
                             {"eas": {"xi1": ("F", "F"), "xi2": ("F", "F")}})

    # Define the shell geometry using a shallow shell approximation
    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, density)

    # Define the shell structure
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
    residue = lambda u, *args: residuo1(u, shell, eas_field, F_ext, nx, ny, nz)
    jacobian = lambda u, *args: jacobian1(u, shell, eas_field, F_ext, nx, ny, nz)
    stability = shell_stability
    output = lambda u, *args: shallow_spherical_panel_output_results(shell, a / 2, b / 2, u, *args)

    # Define boundaries for continuation process
    boundary_continuation = np.zeros((n + p, 2))
    boundary_continuation[:-1, 0] = -1000  # Lower bound for variables
    boundary_continuation[:-1, 1] = 1000  # Upper bound for variables
    boundary_continuation[-1, 0] = -700  # Lower bound for parameter
    boundary_continuation[-1, 1] = 700  # Upper bound for parameter

    # Define the continuation model
    modelo = {'n': n_dof,
              'p': 1,
              'residue': residue,
              'jacobian': jacobian,
              'stability_check': stability,
              'boundary': boundary_continuation,
              'output_function': output}

    # Initialize the continuation process
    continuation = Continuation(modelo)
    continuation.parameters['tol2'] = 0  # Tolerance for convergence
    continuation.parameters['tol1'] = 1E-3  # Step size tolerance
    continuation.parameters['index1'] = -1  # Primary parameter index
    continuation.parameters['index2'] = displacement_field._mapping.index(('u3', 1, 1))  # Index for tracking
    continuation.parameters['h_max'] = 10  # Maximum step size
    continuation.parameters['h0'] = 1e-2  # Maximum step size

    # Define an initial regular point for the continuation process
    u0 = np.zeros(modelo['n'] + modelo['p'])
    u0[-1] = 0.1  # Initial value for the continuation parameter
    H0 = continuation.model['residue'](u0)  # Compute initial residue
    J0 = continuation.model['jacobian'](u0)  # Compute initial Jacobian
    t0 = continuation.tangent_vector(J0)  # Compute initial tangent vector
    w0 = 1  # Step direction

    # Start interactive plotting
    # plt.ion()

    # Perform continuation process
    continuation.continue_branch(u0, t0, w0, 'Branch1')

    # Show the final plot
    plt.show()
