from typing import Any
import numpy as np
from multipledispatch import dispatch

from double_integral_booles_rule import boole_weights_double_integral
from shell import Shell
from shell_loads.shell_conservative_load import ConcentratedForce, PressureLoad


# Function to compute the energy functional for the applied loads on the shell using Koiter's theory
def koiter_load_energy(shell: Shell):
    # Get the number of degrees of freedom (DOF) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Initialize an array to store the energy functional for each DOF
    energy_functional = np.zeros(n_dof)

    # Loop through all DOFs to calculate the energy functional for each one
    for i in range(n_dof):
        energy_functional[i] = koiter_load_energy_density(i, shell.load, shell)

    return energy_functional


# Function to calculate the load energy density for a concentrated force load
@dispatch(int, ConcentratedForce, Shell)
def koiter_load_energy_density(i: int, load, shell):
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


# Function to calculate the load energy density for a pressure load using numerical integration
@dispatch(int, PressureLoad, Shell)
def koiter_load_energy_density(i: int, load, shell):
    # Internal helper function to calculate the energy density for a given pressure load
    def energy_density(pressure, midsurface_geometry1, displacement_expansion1, xi1, xi2):
        # Get the displacement shape function at the current integration points
        U = displacement_expansion1.shape_function(i, xi1, xi2)

        # Compute the reciprocal base vectors at the current integration points
        N1, N2, N3 = midsurface_geometry1.reciprocal_base(xi1, xi2)

        # Compute the displacement field by combining the shape functions and the reciprocal base vectors
        U = U[0] * N1 + U[1] * N2 + U[2] * N3

        # Compute the force due to the pressure (assuming pressure acts along the N3 direction)
        F = pressure * N3

        # Compute the square root of the metric determinant for the mid-surface geometry
        sqrtG = midsurface_geometry1.sqrtG(xi1, xi2)

        # Return the integrand for the energy density using the Einstein summation convention
        return np.einsum('ijxy,ijxy,xy->xy', F, U, sqrtG)

    # Define the lambda function to be used in the integration
    func = lambda xi1, xi2: energy_density(load.pressure, shell.mid_surface_geometry, shell.displacement_expansion, xi1, xi2)

    # Get the integration points and weights for the numerical integration
    xi1, xi2, W = boole_weights_double_integral(shell.mid_surface_domain)

    # Perform the numerical integration and return the negative of the result
    return -np.einsum('xy, xy->', func(xi1, xi2), W)
