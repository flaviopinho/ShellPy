import numpy as np
from multipledispatch import dispatch

from shellpy import Shell
from shellpy.shell_loads import ConcentratedForce, PressureLoad


# Function to compute the energy functional for the applied loads on the shell using Koiter's theory
def fosd_load_energy(shell: Shell):
    # Get the number of degrees of freedom (DOF) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Initialize an array to store the energy functional for each DOF
    energy_functional = np.zeros(n_dof)

    # Loop through all DOFs to calculate the energy functional for each one
    for i in range(n_dof):
        energy_functional[i] = fosd_load_energy_density(i, shell.load, shell)

    return energy_functional


# Function to calculate the load energy density for a concentrated force load
@dispatch(int, ConcentratedForce, Shell)
def fosd_load_energy_density(i: int, load, shell):
    # Get the position where the concentrated force is applied
    position = load.position

    # Get the displacement shape function for the given DOF and position

    UU = shell.displacement_expansion.shape_function(i, position[0], position[1])
    U = UU[0:3]
    V = UU[3:6]

    # Compute the reciprocal base vectors at the given position on the shell's mid-surface
    N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(position[0], position[1])

    # Compute the displacement field by combining the shape functions and the reciprocal base vectors
    U = U[0] * N1 + U[1] * N2 + U[2] * N3

    # Calculate the load energy density as the negative dot product of the load vector and the displacement field
    return -np.dot(np.ravel(load.load_vector), (np.ravel(U)))
