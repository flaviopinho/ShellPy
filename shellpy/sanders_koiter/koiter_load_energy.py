from typing import Any
import numpy as np
from multipledispatch import dispatch

from ..shell import Shell
from ..numeric_integration.boole_integral import boole_weights_simple_integral
from ..numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_y, \
    n_integral_default_z
from ..numeric_integration.integral_weights import double_integral_weights
from ..shell_loads import ConcentratedForceGlobal, PressureLoad
from ..shell_loads import ConcentratedForceLocal


# Function to compute the energy functional for the applied loads on the shell using Koiter's theory
def koiter_load_energy(shell: Shell,
                       n_x=n_integral_default_x,
                       n_y=n_integral_default_y,
                       n_z=n_integral_default_z,
                       integral_method=boole_weights_simple_integral):
    """
    Computes the energy functional associated with the applied loads on the shell.
    This is done using Koiter's theory and numerical integration.

    :param shell: The shell object containing geometry and load information.
    :param n_x, n_y, n_z: Default integral divisions along respective directions.
    :param integral_method: The numerical integration method to be used.
    :return: A numpy array representing the energy functional for each degree of freedom (DOF).
    """
    # Get the number of degrees of freedom (DOF) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Initialize an array to store the energy functional for each DOF
    energy_functional = np.zeros(n_dof)

    # Loop through all DOFs to calculate the energy functional for each one
    for i in range(n_dof):
        energy_functional[i] = koiter_load_energy_density(i, shell.load, shell, n_x, n_y, n_z, integral_method)

    return energy_functional


# Function to calculate the load energy density for a concentrated force load
@dispatch(int, ConcentratedForceGlobal, Shell, object, object, object, object)
def koiter_load_energy_density(i: int, load, shell, *args):
    """
    Computes the energy density contribution due to a concentrated force applied to the shell.

    :param i: Index of the displacement degree of freedom.
    :param load: The concentrated force load applied to the shell.
    :param shell: The shell object containing displacement expansion and geometry.
    :return: The computed energy density value.
    """
    # Get the position where the concentrated force is applied
    position = load.position

    # Get the displacement shape function for the given DOF and position
    U = shell.displacement_expansion.shape_function(i, position[0], position[1])

    # Compute the reciprocal base vectors at the given position on the shell's mid-surface
    N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(position[0], position[1])

    # Compute the displacement field by combining the shape functions and the reciprocal base vectors
    U = U[0] * N1 + U[1] * N2 + U[2] * N3

    # Calculate the load energy density as the negative dot product of the load vector and the displacement field
    return -np.dot(np.ravel(load.load_vector), (np.ravel(U)))


# Function to calculate the load energy density for a concentrated force load
@dispatch(int, ConcentratedForceLocal, Shell, object, object, object, object)
def koiter_load_energy_density(i: int, load, shell, *args):
    """
    Computes the energy density contribution due to a concentrated force applied to the shell.

    :param i: Index of the displacement degree of freedom.
    :param load: The concentrated force load applied to the shell.
    :param shell: The shell object containing displacement expansion and geometry.
    :return: The computed energy density value.
    """
    # Get the position where the concentrated force is applied
    position = load.position

    # Get the displacement shape function for the given DOF and position
    U = shell.displacement_expansion.shape_function(i, position[0], position[1])

    G = shell.mid_surface_geometry.metric_tensor_covariant_components(position[0], position[1])

    P = np.zeros(np.shape(load.load_vector))
    P[0] = load.load_vector[0] / np.sqrt(G[0, 0])
    P[1] = load.load_vector[1] / np.sqrt(G[1, 1])
    P[2] = load.load_vector[2]

    # Calculate the load energy density as the negative dot product of the load vector and the displacement field
    return -np.dot(np.ravel(P), (np.ravel(U)))


# Function to calculate the load energy density for a pressure load using numerical integration
@dispatch(int, PressureLoad, Shell, object, object, object, object)
def koiter_load_energy_density(i: int, load, shell, n_x, n_y, n_z, integral_method):
    """
    Computes the energy density contribution due to a pressure load applied to the shell.
    This is done using numerical integration over the shell's mid-surface.

    :param i: Index of the displacement degree of freedom.
    :param load: The pressure load applied to the shell.
    :param shell: The shell object containing displacement expansion and geometry.
    :param n_x, n_y, n_z: Integral division parameters for numerical integration.
    :param integral_method: The numerical integration method to be used.
    :return: The computed energy density value.
    """

    # Internal helper function to calculate the energy density for a given pressure load
    def energy_density(pressure, midsurface_geometry1, displacement_expansion1, xi1, xi2):
        """
        Computes the energy density at a given integration point for a pressure load.

        :param pressure: The applied pressure value.
        :param midsurface_geometry1: The mid-surface geometry of the shell.
        :param displacement_expansion1: The displacement expansion functions.
        :param xi1, xi2: Integration point coordinates.
        :return: Computed energy density at the given point.
        """
        # Get the displacement shape function 