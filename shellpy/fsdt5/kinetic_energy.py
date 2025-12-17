from time import time

import numpy as np

from shellpy import Shell
from shellpy.materials.shell_density import shell_density
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
from shellpy.numeric_integration.integral_weights import double_integral_weights


def kinetic_energy(shell: Shell, n_x=20, n_y=20, n_z=10, integral_method=gauss_weights_simple_integral):

    # Get integration points and weights for the double integral over the mid-surface domain
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)

    h = shell.thickness(xi1, xi2)

    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    # Shape of xi1 (discretized domain in terms of xi1 and xi2)
    n_xy = np.shape(xi1)
    n_xyz = np.shape(xi1) + (np.shape(xi3)[-1],)

    # Number of degrees of freedom (dof) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Compute the contravariant metric tensor components and sqrt(G) for the shell geometry
    metric_tensor_contravariant_components = shell.mid_surface_geometry.metric_tensor_contravariant_components_extended(
        xi1, xi2)

    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    Wxy1 = sqrtG * Wxy

    # Calculate the constitutive tensor C for the thin shell material
    rho = shell_density(shell.material, xi1, xi2, xi3)

    W0 = 1 / 2 * np.einsum('xyz, xy->xy', rho * xi3 ** 0 * det_shifter_tensor * Wz, Wxy1)
    W1 = 1 / 2 * np.einsum('xyz, xy->xy', rho * xi3 ** 1 * det_shifter_tensor * Wz, Wxy1)
    W2 = 1 / 2 * np.einsum('xyz, xy->xy', rho * xi3 ** 2 * det_shifter_tensor * Wz, Wxy1)

    # Initialize arrays for linear strain components (gamma_lin) and their associated quantities (rho_lin)
    u = np.zeros((n_dof, 3) + n_xy)
    v = np.zeros((n_dof, 3) + n_xy)

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        U = shell.displacement_expansion.shape_function(i, xi1, xi2)
        u[i] = U[0:3]
        v[i, 0:2] = U[3:5]

    print('Calculating quadratic kinetic energy functional...')
    start = time()
    # First term: Contribution from displacement fields only
    kinetic_energy_tensor = np.einsum('ijxy, aixy, bjxy, xy->ab',
                                      metric_tensor_contravariant_components,
                                      u, u, W0, optimize=True)

    # Second term: Contribution from coupling between displacement fields and mu terms
    kinetic_energy_tensor += 2 * np.einsum('ijxy, aixy, bjxy, xy->ab',
                                           metric_tensor_contravariant_components,
                                           u, v, W1, optimize=True)

    # Third term: Contribution from mu terms only
    kinetic_energy_tensor += np.einsum('ijxy, aixy, bjxy, xy->ab',
                                       metric_tensor_contravariant_components,
                                       v, v, W2, optimize=True)
    stop = time()
    print('time= ', stop - start)

    return kinetic_energy_tensor
