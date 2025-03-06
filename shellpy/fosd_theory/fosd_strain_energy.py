from time import time

import numpy as np

from shellpy import Shell, boole_weights_double_integral, boole_weights_triple_integral, boole_weights_simple_integral
from shellpy.fosd_theory.fosd_strain_tensor import fosd_linear_strain_components, fosd_nonlinear_strain_components
from shellpy.koiter_shell_theory import koiter_nonlinear_strain_components_total


def fosd_strain_energy(shell: Shell, n_x, n_y, n_z):
    h = shell.thickness()

    # Get integration points and weights for the double integral over the mid-surface domain
    xi1, xi2, Wxy = boole_weights_double_integral(shell.mid_surface_domain.edges["xi1"],
                                                  shell.mid_surface_domain.edges["xi2"],
                                                  n_x, n_y)

    xi3, Wz = boole_weights_simple_integral((-h / 2, h / 2), n_z)

    # Shape of xi1 (discretized domain in terms of xi1 and xi2)
    n_xy = np.shape(xi1)
    n = np.shape(xi1) + np.shape(xi3)

    # Number of degrees of freedom (dof) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Compute the contravariant metric tensor components and sqrt(G) for the shell geometry
    metric_tensor_contravariant_components = shell.mid_surface_geometry.metric_tensor_contravariant_components_extended(xi1, xi2)

    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    shifter_tensor_inverse = shell.mid_surface_geometry.shifter_tensor_inverse_cubic_approximation(xi1, xi2, xi3)

    Wz0 = 1/2 * np.einsum('z, xyz, z->xyz', xi3 ** 0, det_shifter_tensor, Wz)
    Wz1 = 1/2 * np.einsum('z, xyz, z->xyz', xi3 ** 1, det_shifter_tensor, Wz)
    Wz2 = 1/2 * np.einsum('z, xyz, z->xyz', xi3 ** 2, det_shifter_tensor, Wz)
    Wz3 = 1/2 * np.einsum('z, xyz, z->xyz', xi3 ** 3, det_shifter_tensor, Wz)
    Wz4 = 1/2 * np.einsum('z, xyz, z->xyz', xi3 ** 4, det_shifter_tensor, Wz)

    Wxy1 = sqrtG * Wxy

    metric_tensor2 = np.zeros((3, 3) + n)
    metric_tensor2[0:2, 0:2] = np.einsum('oaxyz, gbxyz, ogxy -> abxyz',
                                         shifter_tensor_inverse,
                                         shifter_tensor_inverse,
                                         metric_tensor_contravariant_components[0:2, 0:2])
    metric_tensor2[2, 2] = 1

    # Calculate the constitutive tensor C for the thin shell material
    C = shell.material.constitutive_tensor(metric_tensor2)

    # Initialize arrays for linear strain components (gamma_lin) and their associated quantities (rho_lin)
    epsilon0_lin = np.zeros((n_dof, 3, 3) + n_xy)
    epsilon1_lin = np.zeros((n_dof, 3, 3) + n_xy)
    epsilon2_lin = np.zeros((n_dof, 3, 3) + n_xy)

    L0_lin = np.zeros((n_dof, 3, 3) + n_xy)
    L1_lin = np.zeros((n_dof, 3, 3) + n_xy)
    L2_lin = np.zeros((n_dof, 3, 3) + n_xy)

    L01_lin = np.zeros((n_dof, 3, 3) + n_xy)
    L02_lin = np.zeros((n_dof, 3, 3) + n_xy)

    L12_lin = np.zeros((n_dof, 3, 3) + n_xy)

    L10_lin = np.zeros((n_dof, 3, 3) + n_xy)
    L20_lin = np.zeros((n_dof, 3, 3) + n_xy)

    L21_lin = np.zeros((n_dof, 3, 3) + n_xy)

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i], epsilon2_lin[i] = fosd_linear_strain_components(shell.mid_surface_geometry,
                                                                                          shell.displacement_expansion,
                                                                                          i, xi1, xi2)

        L0_lin[i] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon0_lin[i], Wz0)
        L1_lin[i] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon1_lin[i], Wz2)
        L2_lin[i] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon2_lin[i], Wz4)

        L01_lin[i] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon0_lin[i], Wz1)
        L02_lin[i] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon0_lin[i], Wz2)
        L12_lin[i] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon1_lin[i], Wz3)

        L10_lin[i] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon1_lin[i], Wz1)
        L20_lin[i] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon2_lin[i], Wz2)
        L21_lin[i] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon2_lin[i], Wz3)

        print(f'Calculating linear components {i} of {n_dof}')

    # Initialize array for nonlinear strain components
    epsilon0_nl = np.zeros((n_dof, n_dof, 3, 3) + n_xy)
    epsilon1_nl = np.zeros((n_dof, n_dof, 3, 3) + n_xy)
    epsilon2_nl = np.zeros((n_dof, n_dof, 3, 3) + n_xy)

    L0_nl = np.zeros((n_dof, n_dof, 3, 3) + n_xy)
    L1_nl = np.zeros((n_dof, n_dof, 3, 3) + n_xy)
    L2_nl = np.zeros((n_dof, n_dof, 3, 3) + n_xy)

    L01_nl = np.zeros((n_dof, n_dof, 3, 3) + n_xy)
    L02_nl = np.zeros((n_dof, n_dof, 3, 3) + n_xy)
    L12_nl = np.zeros((n_dof, n_dof, 3, 3) + n_xy)

    L10_nl = np.zeros((n_dof, n_dof, 3, 3) + n_xy)
    L20_nl = np.zeros((n_dof, n_dof, 3, 3) + n_xy)
    L21_nl = np.zeros((n_dof, n_dof, 3, 3) + n_xy)

    # Loop through the degrees of freedom to compute nonlinear strain components
    for i in range(n_dof):
        for j in range(n_dof):  # Compute only for i <= j to exploit symmetry
            epsilon0_nl[i, j], epsilon1_nl[i, j], epsilon2_nl[i, j] = fosd_nonlinear_strain_components(
                shell.mid_surface_geometry,
                shell.displacement_expansion,
                i, j, xi1, xi2)

            L0_nl[i, j] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon0_nl[i, j], Wz0)
            L1_nl[i, j] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon1_nl[i, j], Wz2)
            L2_nl[i, j] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon2_nl[i, j], Wz4)

            L01_nl[i, j] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon0_nl[i, j], Wz1)
            L02_nl[i, j] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon0_nl[i, j], Wz2)
            L12_nl[i, j] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon1_nl[i, j], Wz3)

            L10_nl[i, j] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon1_nl[i, j], Wz1)
            L20_nl[i, j] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon1_nl[i, j], Wz2)
            L21_nl[i, j] = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon2_nl[i, j], Wz3)

            print(f'Calculating nonlinear components ({i}, {j}) of ({n_dof}, {n_dof})')

    # Calculate the quadratic strain energy functional
    print('Calculating quadratic strain energy functional...')
    start = time()
    quadratic_energy_tensor = np.einsum('mabxy, nabxy, xy->mn', L0_lin, epsilon0_lin, Wxy1,
                                        optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', L1_lin, epsilon1_lin, Wxy1,
                                         optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', L2_lin, epsilon2_lin, Wxy1,
                                         optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', L01_lin, epsilon1_lin, Wxy1,
                                        optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', L02_lin, epsilon2_lin, Wxy1,
                                         optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', L12_lin, epsilon2_lin, Wxy1,
                                         optimize=True)

    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', L10_lin, epsilon0_lin, Wxy1,
                                         optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', L20_lin, epsilon0_lin, Wxy1,
                                         optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', L21_lin, epsilon1_lin, Wxy1,
                                         optimize=True)
    stop = time()
    print('time= ', stop - start)

    print('Calculating cubic strain energy functional...')
    start = time()
    cubic_energy_tensor = 2 * np.einsum('mabxy, noabxy, xy->mno', L0_lin, epsilon0_nl, Wxy1,
                                    optimize=True)
    cubic_energy_tensor += 2 * np.einsum('mabxy, noabxy, xy->mno', L1_lin, epsilon1_nl, Wxy1,
                                     optimize=True)
    cubic_energy_tensor += 2 * np.einsum('mabxy, noabxy, xy->mno', L2_lin, epsilon2_nl, Wxy1,
                                     optimize=True)

    cubic_energy_tensor += 2 * np.einsum('mnabxy, oabxy, xy->mno', L01_nl, epsilon1_lin, Wxy1,
                                        optimize=True)
    cubic_energy_tensor += 2 * np.einsum('mnabxy, oabxy, xy->mno', L02_nl, epsilon2_lin, Wxy1,
                                         optimize=True)
    cubic_energy_tensor += 2 * np.einsum('mnabxy, oabxy, xy->mno', L12_nl, epsilon2_lin, Wxy1,
                                         optimize=True)

    cubic_energy_tensor += 2 * np.einsum('mabxy, noabxy, xy->mno', L10_lin, epsilon0_nl, Wxy1,
                                        optimize=True)
    cubic_energy_tensor += 2 * np.einsum('mabxy, noabxy, xy->mno', L20_lin, epsilon0_nl, Wxy1,
                                         optimize=True)
    cubic_energy_tensor += 2 * np.einsum('mabxy, noabxy, xy->mno', L21_lin, epsilon1_nl, Wxy1,
                                         optimize=True)
    stop = time()
    print('time= ', stop - start)

    print('Calculating quartic strain energy functional...')
    start = time()
    quartic_energy_tensor = np.einsum('mnabxy, opabxy, xy->mnop', L0_nl, epsilon0_nl, Wxy1,
                                      optimize=True)
    quartic_energy_tensor += np.einsum('mnabxy, opabxy, xy->mnop', L1_nl, epsilon1_nl, Wxy1,
                                       optimize=True)
    quartic_energy_tensor += np.einsum('mnabxy, opabxy, xy->mnop', L2_nl, epsilon2_nl, Wxy1,
                                       optimize=True)

    quartic_energy_tensor += np.einsum('mnabxy, opabxy, xy->mnop', L01_nl, epsilon1_nl, Wxy1,
                                      optimize=True)
    quartic_energy_tensor += np.einsum('mnabxy, opabxy, xy->mnop', L02_nl, epsilon2_nl, Wxy1,
                                       optimize=True)
    quartic_energy_tensor += np.einsum('mnabxy, opabxy, xy->mnop', L12_nl, epsilon2_nl, Wxy1,
                                       optimize=True)

    quartic_energy_tensor += np.einsum('mnabxy, opabxy, xy->mnop', L10_nl, epsilon0_nl, Wxy1,
                                       optimize=True)
    quartic_energy_tensor += np.einsum('mnabxy, opabxy, xy->mnop', L20_nl, epsilon0_nl, Wxy1,
                                       optimize=True)
    quartic_energy_tensor += np.einsum('mnabxy, opabxy, xy->mnop', L21_nl, epsilon1_nl, Wxy1,
                                       optimize=True)

    stop = time()
    print('time= ', stop - start)

    return quadratic_energy_tensor, cubic_energy_tensor, quartic_energy_tensor


def fosd_quadratic_strain_energy(shell: Shell, n_x, n_y, n_z):
    h = shell.thickness()

    # Get integration points and weights for the double integral over the mid-surface domain
    xi1, xi2, Wxy = boole_weights_double_integral(shell.mid_surface_domain.edges["xi1"],
                                                  shell.mid_surface_domain.edges["xi2"],
                                                  n_x, n_y)

    xi3, Wz = boole_weights_simple_integral((-h / 2, h / 2), n_z)

    # Shape of xi1 (discretized domain in terms of xi1 and xi2)
    n_xy = np.shape(xi1)
    n = np.shape(xi1) + np.shape(xi3)

    # Number of degrees of freedom (dof) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Compute the contravariant metric tensor components and sqrt(G) for the shell geometry
    metric_tensor = shell.mid_surface_geometry.metric_tensor_contravariant_components_extended(xi1, xi2)

    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    shifter_tensor_inverse = shell.mid_surface_geometry.shifter_tensor_inverse_cubic_approximation(xi1, xi2, xi3)

    Wz1 = np.einsum('z, xyz, z->xyz', xi3 ** 0, det_shifter_tensor, Wz)
    Wz2 = np.einsum('z, xyz, z->xyz', xi3 ** 2, det_shifter_tensor, Wz)
    Wz3 = np.einsum('z, xyz, z->xyz', xi3 ** 4, det_shifter_tensor, Wz)

    Wxy1 = sqrtG * Wxy

    metric_tensor2 = np.zeros((3, 3) + n)
    metric_tensor2[0:2, 0:2] = np.einsum('oaxyz, gbxyz, ogxy -> abxyz',
                                         shifter_tensor_inverse,
                                         shifter_tensor_inverse,
                                         metric_tensor[0:2, 0:2])
    metric_tensor2[2, 2] = 1

    # Calculate the constitutive tensor C for the thin shell material
    C = shell.material.constitutive_tensor(metric_tensor2)

    # Initialize arrays for linear strain components (gamma_lin) and their associated quantities (rho_lin)
    epsilon0_lin = np.zeros((n_dof, 3, 3) + n_xy)
    epsilon1_lin = np.zeros((n_dof, 3, 3) + n_xy)
    epsilon2_lin = np.zeros((n_dof, 3, 3) + n_xy)

    L0_lin = np.zeros((n_dof, 3, 3) + n_xy)
    L1_lin = np.zeros((n_dof, 3, 3) + n_xy)
    L2_lin = np.zeros((n_dof, 3, 3) + n_xy)

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i], epsilon2_lin[i] = fosd_linear_strain_components(shell.mid_surface_geometry,
                                                                                          shell.displacement_expansion,
                                                                                          i, xi1, xi2)

        L0_lin = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon0_lin, Wz1)
        L1_lin = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon1_lin, Wz2)
        L2_lin = np.einsum('ijklxyz, klxy, xyz->ijxy', C, epsilon1_lin, Wz3)
        print(f'Calculating linear components {i} of {n_dof}')

    # Calculate the quadratic strain energy functional
    print('Calculating quadratic strain energy functional...')
    start = time()
    quadratic_energy_tensor = np.einsum('mabxy, nabxy, xy->mn', L0_lin, epsilon0_lin, Wxy1,
                                        optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', L1_lin, epsilon1_lin, Wxy1,
                                         optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', L2_lin, epsilon2_lin, Wxy1,
                                         optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Return the computed strain energy tensors for quadratic terms
    return quadratic_energy_tensor
