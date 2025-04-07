from time import time

import numpy as np

from shellpy import Shell
from shellpy.materials.shell_density import shell_density
from shellpy.numeric_integration.boole_integral import boole_weights_simple_integral
from shellpy.numeric_integration.integral_weights import double_integral_weights


def fosd_kinetic_energy(shell: Shell, n_x, n_y, n_z, integral_method=boole_weights_simple_integral):
    h = shell.thickness()

    # Get integration points and weights for the double integral over the mid-surface domain
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)

    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    # Shape of xi1 (discretized domain in terms of xi1 and xi2)
    n_xy = np.shape(xi1)
    n_xyz = np.shape(xi1) + np.shape(xi3)

    # Number of degrees of freedom (dof) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Compute the contravariant metric tensor components and sqrt(G) for the shell geometry
    metric_tensor_contravariant_components = shell.mid_surface_geometry.metric_tensor_contravariant_components_extended(
        xi1, xi2)

    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    shifter_tensor_inverse = shell.mid_surface_geometry.shifter_tensor_inverse_approximation(xi1, xi2, xi3)

    Wxy1 = sqrtG * Wxy

    metric_tensor2 = np.zeros((3, 3) + n_xyz)
    metric_tensor2[0:2, 0:2] = np.einsum('oaxyz, gbxyz, ogxy -> abxyz',
                                         shifter_tensor_inverse,
                                         shifter_tensor_inverse,
                                         metric_tensor_contravariant_components[0:2, 0:2])
    metric_tensor2[2, 2] = 1

    # Calculate the constitutive tensor C for the thin shell material
    rho = shell_density(shell.material, xi1, xi2, xi3)

    W0 = 1 / 2 * np.einsum('xyz, z, xyz, z, xy->xy', rho, xi3 ** 0, det_shifter_tensor, Wz, Wxy1)
    W1 = 1 / 2 * np.einsum('xyz, z, xyz, z, xy->xy', rho, xi3 ** 1, det_shifter_tensor, Wz, Wxy1)
    W2 = 1 / 2 * np.einsum('xyz, z, xyz, z, xy->xy', rho, xi3 ** 2, det_shifter_tensor, Wz, Wxy1)

    # Initialize arrays for linear strain components (gamma_lin) and their associated quantities (rho_lin)
    u = np.zeros((n_dof, 3) + n_xy)
    v = np.zeros((n_dof, 3) + n_xy)

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        u[i], v[i] = shell.displacement_expansion.shape_function(i, xi1, xi2)



    print('Calculating quadratic kinetic energy functional...')
    start = time()
    kinetic_energy_tensor = np.einsum('omxy, oxy, nxy, xy->mn',
                                      metric_tensor_contravariant_components,
                                      u, u, W0,
                                      optimize=True)
    kinetic_energy_tensor += 2 * np.einsum('omxy, oxy, nxy, xy->mn',
                                          metric_tensor_contravariant_components,
                                          u, v, W1,
                                          optimize=True)
    kinetic_energy_tensor += np.einsum('omxy, oxy, nxy, xy->mn',
                                       metric_tensor_contravariant_components,
                                       v, v, W2,
                                       optimize=True)
    stop = time()
    print('time= ', stop - start)

    return kinetic_energy_tensor
