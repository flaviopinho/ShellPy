from time import time
import numpy as np
from shellpy import Shell
from shellpy.fosd_theory.fosd_strain_tensor import fosd_linear_strain_components
from shellpy.fosd_theory2.fosd2_strain_vector import fosd2_linear_strain_vector
from shellpy.materials.constitutive_matrix_fosd2 import constitutive_matrix_for_fosd2
from shellpy.numeric_integration.boole_integral import boole_weights_simple_integral
from shellpy.numeric_integration.integral_weights import double_integral_weights


def fosd2_quadratic_strain_energy(shell: Shell, n_x, n_y, n_z, integral_method=boole_weights_simple_integral):


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

    shifter_tensor_inverse = shell.mid_surface_geometry.shifter_tensor_inverse_approximation(xi1, xi2, xi3)

    Wxy1 = sqrtG * Wxy

    metric_tensor2 = np.zeros((3, 3) + n_xyz)
    metric_tensor2[0:2, 0:2] = np.einsum('oaxyz, gbxyz, ogxy -> abxyz',
                                         shifter_tensor_inverse,
                                         shifter_tensor_inverse,
                                         metric_tensor_contravariant_components[0:2, 0:2])
    metric_tensor2[2, 2] = 1

    # Calculate the constitutive tensor C for the thin shell material
    C = constitutive_matrix_for_fosd2(shell.mid_surface_geometry, shell.material, xi1, xi2, xi3)

    C0 = 1 / 2 * np.einsum('ijxyz, xyz, xyz, xyz->ijxy', C, xi3 ** 0, det_shifter_tensor, Wz)
    C1 = 1 / 2 * np.einsum('ijxyz, xyz, xyz, xyz->ijxy', C, xi3 ** 1, det_shifter_tensor, Wz)
    C2 = 1 / 2 * np.einsum('ijxyz, xyz, xyz, xyz->ijxy', C, xi3 ** 2, det_shifter_tensor, Wz)
    C3 = 1 / 2 * np.einsum('ijxyz, xyz, xyz, xyz->ijxy', C, xi3 ** 3, det_shifter_tensor, Wz)
    C4 = 1 / 2 * np.einsum('ijxyz, xyz, xyz, xyz->ijxy', C, xi3 ** 4, det_shifter_tensor, Wz)

    # Initialize arrays for linear strain components (gamma_lin) and their associated quantities (rho_lin)
    epsilon0_lin = np.zeros((n_dof, 6) + n_xy)
    epsilon1_lin = np.zeros((n_dof, 6) + n_xy)
    epsilon2_lin = np.zeros((n_dof, 6) + n_xy)

    L0_lin = np.zeros((n_dof, 6) + n_xy)
    L1_lin = np.zeros((n_dof, 6) + n_xy)
    L2_lin = np.zeros((n_dof, 6) + n_xy)

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i], epsilon2_lin[i] = fosd2_linear_strain_vector(shell.mid_surface_geometry,
                                                                                       shell.displacement_expansion,
                                                                                       i, xi1, xi2)

        L0_lin[i] = (np.einsum('ijxy, jxy->ixy', C0, epsilon0_lin[i], optimize=True) +
                     np.einsum('ijxy, jxy->ixy', C1, epsilon1_lin[i], optimize=True) +
                     np.einsum('ijxy, jxy->ixy', C2, epsilon2_lin[i], optimize=True))

        L1_lin[i] = (np.einsum('ijxy, jxy->ixy', C1, epsilon0_lin[i], optimize=True) +
                     np.einsum('ijxy, jxy->ixy', C2, epsilon1_lin[i], optimize=True) +
                     np.einsum('ijxy, jxy->ixy', C3, epsilon2_lin[i], optimize=True))

        L2_lin[i] = (np.einsum('ijxy, jxy->ixy', C2, epsilon0_lin[i], optimize=True) +
                     np.einsum('ijxy, jxy->ixy', C3, epsilon1_lin[i], optimize=True) +
                     np.einsum('ijxy, jxy->ixy', C4, epsilon2_lin[i], optimize=True))

        print(f'Calculating linear components {i} of {n_dof}')

    # Calculate the quadratic strain energy functional
    print('Calculating quadratic strain energy functional...')
    start = time()
    quadratic_energy_tensor = np.einsum('maxy, naxy, xy->mn', L0_lin, epsilon0_lin, Wxy1,
                                        optimize=True)
    quadratic_energy_tensor += np.einsum('maxy, naxy, xy->mn', L1_lin, epsilon1_lin, Wxy1,
                                         optimize=True)
    quadratic_energy_tensor += np.einsum('maxy, naxy, xy->mn', L2_lin, epsilon2_lin, Wxy1,
                                         optimize=True)
    stop = time()
    print('time= ', stop - start)

    return quadratic_energy_tensor
