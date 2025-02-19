"""
This example analyzes the nonlinear behavior of a doubly curved shell
(a spherical panel) under a concentrated central load.  This shell has been
previously studied by Kobayashi and Leissa (https://doi.org/10.1016/0020-7462(94)00030-E),
Amabili (https://doi.org/10.1016/j.ijnonlinmec.2004.08.007),
and Pinho et al. (DOI: 10.1016/j.engstruct.2021.113674).  This script determines
the nonlinear static response.
"""

import sys
import matplotlib.pyplot as plt
import sympy as sym
import numpy as np

# Import custom modules related to shell analysis
from shellpy.displacement_expansion import simply_supported  # Defines simply supported boundary conditions
from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion # Eigenfunction expansion for displacements
from shellpy.expansions.polinomial_expansion import GenericPolynomialSeries # Polynomial series expansion
from shellpy.mid_surface_domain import RectangularMidSurfaceDomain # Defines the geometry of the mid-surface
from shellpy.koiter_shell_theory import fast_koiter_strain_energy # Computes Koiter's strain energy
from shellpy.tensor_derivatives import tensor_derivative # For calculating tensor derivatives (likely for strain)
from shellpy.koiter_shell_theory.koiter_load_energy import koiter_load_energy # Computes work done by external loads
from shellpy.shell_loads.shell_conservative_load import ConcentratedForce # Defines a concentrated force
from shellpy.material import LinearElasticMaterial # Defines a linear elastic material model
from shellpy.shell import Shell # Represents the shell structure
from shellpy.thickness import ConstantThickness # Defines constant thickness
from shellpy.midsurface_geometry import MidSurfaceGeometry, xi1_, xi2_ # Defines the mid-surface geometry and coordinate system

# Add the path to the ContinuationPy library.  This is assumed to be in a
# relative directory '../../../ContinuationPy/ContinuationPy'.
sys.path.append('../../ContinuationPy/ContinuationPy')
import continuation # Imports the continuation library for solving nonlinear equations.


def amabili_shallow_shell_residue(F_int, F_ext, x, *args):
    """
    Calculates the residual for the nonlinear equilibrium equations.

    Args:
        F_int: Internal force vector (from strain energy).
        F_ext: External force vector.
        x: Vector of unknowns, containing displacement coefficients (u) and load parameter (p).
        *args: Additional arguments (not used here).

    Returns:
        The residual vector.
    """
    u = x[:-1]  # Extract displacement coefficients from x
    p = x[-1]   # Extract load parameter from x
    # Calculate the total internal force vector, accounting for nonlinear terms.
    F_int_tot = np.einsum('ij, j->i', F_int[0], u) + \
                np.einsum('ijk, j, k->i', F_int[1], u, u) + \
                np.einsum('ijkl, j, k, l->i', F_int[2], u, u, u)
    return F_int_tot + F_ext * p # Return the residual: F_int_tot + p*F_ext = 0 at equilibrium


def amabili_shallow_shell_jacobian(J_int, F_ext, x, *args):
    """
    Calculates the Jacobian of the residual with respect to the unknowns.

    Args:
        J_int: Jacobian of the internal force vector with respect to displacements.
        F_ext: External force vector.
        x: Vector of unknowns (displacement coefficients and load parameter).
        *args: Additional arguments (not used here).

    Returns:
        The Jacobian matrix.
    """
    u = x[:-1]  # Extract displacement coefficients
    # p = x[-1]  # Load parameter (not directly used in Jacobian calculation here)
    J_int_tot = J_int[0] + \
                np.einsum('ijk, k->ij', J_int[1], u) + \
                np.einsum('ijkl, k, l->ij', J_int[2], u, u)

    return np.hstack((J_int_tot, F_ext[:, np.newaxis])) # Combine internal Jacobian and external force vector
                                                          # to form the complete Jacobian.


def amabili_shallow_shell_stability(u, J, model, *args):
    """
    Analyzes the stability of the shell at a given solution point.

    Args:
        u: Displacement coefficients.
        J: Jacobian matrix of the residual equations.
        model: Dictionary containing model parameters (including 'n' for the number of degrees of freedom).
        *args: Additional arguments (not used here).

    Returns:
        stability: The maximum real part of the eigenvalues of the Jacobian (Jx).
        tipo: String indicating the type of bifurcation ('H' for Hopf, 'SN' for Saddle-Node,
              'PR' for Regular Point, 'BC' for Complex Bifurcation).
    """
    # Extract the Jx submatrix, corresponding to the degrees of freedom.  This assumes
    # the Jacobian is structured such that the top-left n x n block corresponds to the
    # displacement degrees of freedom.
    Jx = -J[:model['n'], :model['n']]

    # Calculate the eigenvalues of Jx. These eigenvalues determine the stability
    # of the equilibrium point.
    eigen_values = np.linalg.eigvals(Jx)

    # Separate the real and imaginary parts of the eigenvalues.
    real_part = np.real(eigen_values)
    imaginary_part = np.imag(eigen_values)

    # The stability is determined by the maximum real part of the eigenvalues.
    # A positive maximum real part indicates instability.
    stability = np.max(real_part)

    tipo = None  # Initialize the bifurcation type.

    # Determine the type of bifurcation based on the eigenvalues.
    index_real_positivo = real_part > 0  # Indices of eigenvalues with positive real parts.
    index_real_negativo = real_part < 0  # Indices of eigenvalues with negative real parts.
    num_real_positivo = np.sum(index_real_positivo) # Number of eigenvalues with positive real parts.
    num_real_negativo = np.sum(index_real_negativo) # Number of eigenvalues with negative real parts.

    if num_real_positivo == 2 and np.any(imaginary_part[index_real_positivo] != 0):
        tipo = 'H'  # Hopf bifurcation: Two complex conjugate eigenvalues cross the imaginary axis.
    elif num_real_positivo == 1 and num_real_negativo >= 0:
        tipo = 'SN'  # Saddle-Node bifurcation: A single real eigenvalue crosses zero.
    elif num_real_positivo == 0 and num_real_negativo >= 0:
        tipo = 'PR'  # Regular Point: All eigenvalues have negative real parts (stable).
    else:
        tipo = 'BC'  # Complex Bifurcation:  Other cases (more complex eigenvalue configurations).

    return stability, tipo


def amabili_output_results(shell, xi1, xi2, x, *args):
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

    return p, U[2], "F", "u_3(a/4,b/4)"  # Return load parameter, center displacement, and labels.


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
    x, y = np.meshgrid(xi1, xi2, indexing='xy')

    # Calculate the deformed shape (mode) using the displacement expansion.
    mode = shell.displacement_expansion(u, x, y)  # Compute mode shape

    # Calculate the original (undeformed) mid-surface geometry.
    z = shell.mid_surface_geometry(x, y)  # Compute deformed geometry

    # Create the plot figure and axes.
    fig = plt.figure(1)
    n = len(fig.axes)
    if n == 1: # If it is the first time the figure is created
        fig.clf() # Clear figure
        ax = fig.add_subplot(1, 2, 1) # First subplot (not used)
        ax = fig.add_subplot(1, 2, 2, projection='3d') # Second subplot (3D plot)
    else: # If figure already exists
        ax = plt.subplot(1, 2, 2) # Select the 3D subplot

    ax.cla()  # Clear the axes.

    # Create a colormap for visualization.
    scmap = plt.cm.ScalarMappable(cmap='jet')  # Define colormap

    # Plot the deformed shell surface. The displacement is scaled by a factor of 5 for visualization.
    ax.plot_surface(z[0, 0] + mode[0] * 5, z[1, 0] + mode[1] * 5, z[2, 0] + mode[2] * 5,
                    facecolors=scmap.to_rgba(mode[2]), # Color based on transverse displacement
                    edgecolor='black', # Black edges
                    linewidth=0.5)  # Edge linewidth

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

    # Define external concentrated force at the center of the shell
    load = ConcentratedForce(0, 0, -1, a / 2, b / 2)

    # Define the rectangular domain representing the mid-surface of the shell
    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    # Define the expansion size for displacement fields
    expansion_size = {"u1": (3, 3),  # Number of modes for in-plane displacement u1
                      "u2": (3, 3),  # Number of modes for in-plane displacement u2
                      "u3": (2, 2)}  # Number of modes for transverse displacement u3

    # Set boundary conditions (simply supported case)
    boundary_conditions = simply_supported

    # Define mode mappings for the displacement field expansion
    mapping = []
    modes_xi1 = [2, 3, 6, 7, 10, 11]
    modes_xi2 = [1, 3, 5, 7, 9, 11]
    for i in range(expansion_size["u1"][0]):
        for j in range(expansion_size["u1"][1]):
            mapping.append(("u1", modes_xi1[i], modes_xi2[j]))

    modes_xi1 = [1, 3, 5, 7, 9, 11]
    modes_xi2 = [2, 3, 6, 7, 10, 11]
    for i in range(expansion_size["u2"][0]):
        for j in range(expansion_size["u2"][1]):
            mapping.append(("u2", modes_xi1[i], modes_xi2[j]))

    modes_xi1 = [1, 3, 5, 7, 9, 11]
    modes_xi2 = [1, 3, 5, 7, 9, 11]
    for i in range(expansion_size["u3"][0]):
        for j in range(expansion_size["u3"][1]):
            mapping.append(("u3", modes_xi1[i], modes_xi2[j]))

    # Define the displacement field expansion using eigenfunctions
    displacement_field = EigenFunctionExpansion(expansion_size, rectangular_domain, boundary_conditions, mapping)

    # Define the shell geometry using a shallow shell approximation
    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = LinearElasticMaterial(E, nu, density)

    # Define the shell structure
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, load)

    # Number of degrees of freedom
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Compute external and internal strain energy using Koiter's method
    U_ext = koiter_load_energy(shell)
    U2_int, U3_int, U4_int = fast_koiter_strain_energy(shell)

    # Number of variables (degrees of freedom)
    n = displacement_field.number_of_degrees_of_freedom()
    # Number of parameters (external loading factor)
    p = 1

    # Compute derivatives of the energy functions
    F_ext = tensor_derivative(U_ext, 0)
    F2_int = tensor_derivative(U2_int, 0)
    F3_int = tensor_derivative(U3_int, 0)
    F4_int = tensor_derivative(U4_int, 0)

    # Compute Jacobian matrices
    J2_int = tensor_derivative(F2_int, 1)
    J3_int = tensor_derivative(F3_int, 1)
    J4_int = tensor_derivative(F4_int, 1)

    # Define functions for residual, Jacobian, stability, and output
    residue = lambda u, *args: amabili_shallow_shell_residue((F2_int, F3_int, F4_int), F_ext, u, *args)
    jacobian = lambda u, *args: amabili_shallow_shell_jacobian((J2_int, J3_int, J4_int), F_ext, u, *args)
    stability = amabili_shallow_shell_stability
    output = lambda u, *args: amabili_output_results(shell, a / 2, b / 2, u, *args)

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
    continuation = continuation.Continuation(modelo)
    continuation.parameters['tol2'] = 0  # Tolerance for convergence
    continuation.parameters['tol1'] = 1E-5  # Step size tolerance
    continuation.parameters['index1'] = -1  # Primary parameter index
    continuation.parameters['index2'] = mapping.index(('u3', 1, 1))  # Index for tracking
    continuation.parameters['h_max'] = 10  # Maximum step size

    # Define an initial regular point for the continuation process
    u0 = np.zeros(modelo['n'] + modelo['p'])
    u0[-1] = 0.1  # Initial value for the continuation parameter
    H0 = continuation.model['residue'](u0)  # Compute initial residue
    J0 = continuation.model['jacobian'](u0)  # Compute initial Jacobian
    t0 = continuation.tangent_vector(J0)  # Compute initial tangent vector
    w0 = 1  # Step direction

    # Start interactive plotting
    #plt.ion()

    # Perform continuation process
    continuation.continue_branch(u0, t0, w0, 'Branch1')

    # Show the final plot
    plt.show()

