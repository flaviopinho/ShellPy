import numpy as np

from shellpy import boole_weights_double_integral
from shellpy import Shell


def fast_koiter_kinetic_energy(shell: Shell, integral_weights=boole_weights_double_integral):
    """
    Calculates the kinetic energy of a shell structure using the Koiter approximation.
    This function computes the kinetic energy tensor based on the displacement fields,
    material properties, and geometric properties of the shell.

    Parameters:
    - shell (Shell): The shell object containing all necessary properties, such as
      material properties, thickness, displacement expansions, and geometric data.
    - integral_weights (function): A function to calculate the integration weights and points for the domain.
      Defaults to 'boole_weights_double_integral'.

    Returns:
    - kinetic_energy_tensor (ndarray): The kinetic energy tensor for the shell.
    """

    # Get the weights (xi1, xi2) and corresponding weights (W) for the double integral
    # The weights are used in numerical integration for the domain of the shell's mid-surface
    xi1, xi2, W = integral_weights(shell.mid_surface_domain)

    # Get the shape of xi1 (which represents a discretized domain in terms of xi1 and xi2)
    n = np.shape(xi1)

    # Get the number of degrees of freedom (dof) for the displacement expansion of the shell
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Retrieve the density of the shell material
    rho = shell.material.density

    # Initialize an array to hold the displacement fields for each degree of freedom and each spatial point
    # Shape is (dof, 3 directions, xi1, xi2)
    displacement_fields = np.zeros((n_dof, 3, n[0], n[1]))

    # Loop through the degrees of freedom and calculate the displacement fields
    for i in range(n_dof):
        # Get the displacement field for degree of freedom i at the points defined by xi1, xi2
        displacement_fields[i] = shell.displacement_expansion.shape_function(i, xi1, xi2)

    # Initialize the metric tensor G with dimensions (3, 3, xi1, xi2)
    G = np.zeros((3, 3, n[0], n[1]))

    # Set the contravariant components of the metric tensor (first two rows and columns) based on the shell geometry
    G[0:2, 0:2] = shell.mid_surface_geometry.metric_tensor_contravariant_components(xi1, xi2)

    # Set the G[2, 2] component to 1, as it's used for the thickness direction
    G[2, 2] = 1

    # Calculate the square root of the determinant of the metric tensor (sqrtG) for the shell geometry
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    # Calculate the constitutive tensor C for the thin shell material
    C = shell.material.thin_shell_constitutive_tensor(G)

    # Get the thickness of the shell at each point in the domain
    h = shell.thickness(xi1, xi2)

    # Compute the kinetic energy tensor using the displacement fields, material properties,
    # geometry, and weights (W) for numerical integration
    # np.einsum is used for efficient summation of products of arrays in specific axes.
    # The result is the kinetic energy tensor, accounting for the material density, thickness,
    # displacement fields, and shell geometry.
    kinetic_energy_tensor = (rho * h / 2) * np.einsum('ijxy, aixy, bjxy, xy, xy->ab', G, displacement_fields,
                                                      displacement_fields, sqrtG, W, optimize=True)

    # Return the computed kinetic energy tensor
    return kinetic_energy_tensor
