from time import time
import numpy as np
from shellpy import Shell
from shellpy.fsdt6.shear_correction_factor import shear_correction_factor
from shellpy.fsdt6.constitutive_matrix_in_shell_frame import constitutive_matrix_in_shell_frame
from shellpy.fsdt5.constitutive_matriz_in_material_frame import constitutive_matrix_in_material_frame
from shellpy.fsdt5.strain_vector import linear_strain_vector
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
from shellpy.numeric_integration.integral_weights import double_integral_weights


def quadratic_strain_energy(shell: Shell, n_x=20, n_y=20, n_z=10, integral_method=gauss_weights_simple_integral):
    # Get integration points and weights for the double integral over the mid-surface domain
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)

    h = shell.thickness(xi1, xi2)

    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    # Shape of xi1 (discretized domain in terms of xi1 and xi2)
    n_xy = np.shape(xi1)
    n_xyz = np.shape(xi1) + (np.shape(xi3)[-1],)

    # Number of degrees of freedom (dof) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    Wxy1 = sqrtG * Wxy

    # Calculate the constitutive tensor C for the thin shell material
    C_material = constitutive_matrix_in_material_frame(shell.mid_surface_geometry, shell.material, (xi1, xi2, xi3))

    if C_material.ndim == 2:
        C_material = np.einsum('ij, xyz->ijxyz', C_material, xi3 ** 0)
    kappa_x, kappa_y, kappa_xy = shear_correction_factor(C_material, xi3, Wz, det_shifter_tensor)

    print('kappa_x = ', kappa_x.mean())
    print('kappa_y = ', kappa_y.mean())
    print('kappa_xy = ', kappa_xy.mean())

    C_material[4, 4] = np.einsum('xyz, xy->xyz', C_material[4, 4], kappa_x)
    C_material[3, 3] = np.einsum('xyz, xy->xyz', C_material[3, 3], kappa_y)
    C_material[3, 4] = np.einsum('xyz, xy->xyz', C_material[3, 4], kappa_xy)
    C_material[4, 3] = np.einsum('xyz, xy->xyz', C_material[4, 3], kappa_xy)

    C = constitutive_matrix_in_shell_frame(shell.mid_surface_geometry, C_material, (xi1, xi2, xi3))

    C0 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 0 * det_shifter_tensor * Wz)
    C1 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 1 * det_shifter_tensor * Wz)
    C2 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 2 * det_shifter_tensor * Wz)
    C3 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 3 * det_shifter_tensor * Wz)
    C4 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 4 * det_shifter_tensor * Wz)

    # Initialize arrays for linear strain components (gamma_lin) and their associated quantities (rho_lin)
    epsilon0_lin = np.zeros((n_dof, 6) + n_xy)
    epsilon1_lin = np.zeros((n_dof, 6) + n_xy)
    epsilon2_lin = np.zeros((n_dof, 6) + n_xy)

    L0_lin = np.zeros((n_dof, 6) + n_xy)
    L1_lin = np.zeros((n_dof, 6) + n_xy)
    L2_lin = np.zeros((n_dof, 6) + n_xy)

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i], epsilon2_lin[i] = linear_strain_vector(shell.mid_surface_geometry,
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

        print(f'Calculating linear components {i + 1} of {n_dof}')

    print(f'Calculating enhanced assumed strain')

    print('Calculating quadratic strain energy functional...')
    start = time()
    strain_energy = 0.5 * np.einsum('maxy, naxy, xy->mn', L0_lin, epsilon0_lin, Wxy1,
                              optimize=True)
    strain_energy += 0.5 * np.einsum('maxy, naxy, xy->mn', L1_lin, epsilon1_lin, Wxy1,
                               optimize=True)
    strain_energy += 0.5 * np.einsum('maxy, naxy, xy->mn', L2_lin, epsilon2_lin, Wxy1,
                               optimize=True)

    stop = time()
    print('time= ', stop - start)

    return strain_energy


