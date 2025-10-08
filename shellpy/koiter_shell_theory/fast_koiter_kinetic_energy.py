import numpy as np

from shellpy import displacement_covariant_derivatives
from shellpy import Shell
from shellpy.materials.shell_density import shell_density
from shellpy.numeric_integration.boole_integral import boole_weights_simple_integral
from shellpy.numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_y, \
    n_integral_default_z
from shellpy.numeric_integration.integral_weights import double_integral_weights


def fast_koiter_kinetic_energy(shell: Shell,
                               n_x=n_integral_default_x,
                               n_y=n_integral_default_y,
                               n_z=n_integral_default_z,
                               integral_method=boole_weights_simple_integral):
    """
    Calculates the kinetic energy of a shell structure using the Koiter approximation.
    This function computes the kinetic energy tensor based on the displacement fields,
    material properties, and geometric properties of the shell.

    Parameters:
    - shell (Shell): The shell object containing all necessary properties, such as
      material properties, thickness, displacement expansions, and geometric data.
    - n_x (int): Number of integration points along the xi1 direction.
    - n_y (int): Number of integration points along the xi2 direction.
    - n_z (int): Number of integration points along the thickness direction.
    - integral_method (function): A function to calculate the integration weights and points for the domain.
      Defaults to 'boole_weights_simple_integral'.

    Returns:
    - kinetic_energy_tensor (ndarray): The kinetic energy tensor for the shell.
    """

    # Get integration points and weights for the double integral over the mid-surface domain
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)

    # Get the thickness of the shell at each integration point
    h = shell.thickness(xi1, xi2)

    # Integration points and weights along the thickness direction (xi3)
    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    # Get the shape of the xi1 array (discretized domain in xi1-xi2 space)
    n = np.shape(xi1)

    # Number of degrees of freedom (DOFs) in the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Compute the contravariant metric tensor components and sqrt(G) for the shell geometry
    G = shell.mid_surface_geometry.metric_tensor_contravariant_components_extended(xi1, xi2)
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    # Compute the determinant of the shifter tensor at each point
    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    # Retrieve the density of the shell material at each integration point
    rho = shell_density(shell.material, xi1, xi2, xi3)

    # Multiply surface integration weights by sqrt(G) for proper scaling
    Wxy = sqrtG * Wxy

    # Compute weighted integral coefficients used in kinetic energy calculations
    W0 = 1 / 2 * np.einsum('xyz, xy, xyz, xyz, xyz->xy', rho, Wxy, xi3 ** 0, det_shifter_tensor, Wz)
    W1 = 1 / 2 * np.einsum('xyz, xy, xyz, xyz, xyz->xy', rho, Wxy, xi3 ** 1, det_shifter_tensor, Wz)
    W2 = 1 / 2 * np.einsum('xyz, xy, xyz, xyz, xyz->xy', rho, Wxy, xi3 ** 2, det_shifter_tensor, Wz)

    # Initialize an array to hold the displacement fields for each degree of freedom and spatial point
    # Shape: (n_dof, 3 spatial directions, xi1 grid size, xi2 grid size)
    displacement_fields = np.zeros((n_dof, 3, n[0], n[1]))

    # Initialize an array to store the linear components of the third moment tensor m^3 = mu_i M^i
    # Shape: (n_dof, 3 spatial directions, xi1 grid size, xi2 grid size)
    mu = np.zeros((n_dof, 3, n[0], n[1]))

    # Compute displacement fields and associated moment terms for each degree of freedom
    for i in range(n_dof):
        # Compute the displacement field associated with the ith shape function at each integration point
        displacement_fields[i] = shell.displacement_expansion.shape_function(i, xi1, xi2)

        # Compute the covariant derivatives of the displacement field
        dcu, _ = displacement_covariant_derivatives(shell.mid_surface_geometry,
                                                    shell.displacement_expansion, i, xi1, xi2)

        # Convert covariant derivatives to contravariant form using the metric tensor
        dcu_contra = np.einsum('mi...,i...->m...', G, dcu)

        # Compute components of the linearized curvature term mu
        mu[i, 0] = -dcu_contra[2, 0]  # First component
        mu[i, 1] = -dcu_contra[2, 1]  # Second component
        mu[i, 2] = dcu_contra[0, 0] + dcu_contra[1, 1]  # Third component (sum of normal strains)

    # Compute the kinetic energy tensor using the computed displacement fields, metric tensor, and weights
    # np.einsum is used for efficient tensor contractions

    # First term: Contribution from displacement fields only
    kinetic_energy_tensor = np.einsum('ijxy, aixy, bjxy, xy->ab',
                                      G, displacement_fields, displacement_fields, W0, optimize=True)

    # Second term: Contribution from coupling between displacement fields and mu terms
    kinetic_energy_tensor += 2 * np.einsum('ijxy, aixy, bjxy, xy->ab',
                                           G, displacement_fields, mu, W1, optimize=True)

    # Third term: Contribution from mu terms only
    kinetic_energy_tensor += np.einsum('ijxy, aixy, bjxy, xy->ab',
                                       G, mu, mu, W2, optimize=True)

    # Return the computed kinetic energy tensor
    return kinetic_energy_tensor

