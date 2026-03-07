import numpy as np
from shellpy import Shell, cache_function
from shellpy.fsdt7_eas.enhanced_assumed_strain import enhanced_assumed_strain
from shellpy.fsdt6.shear_correction_factor import shear_correction_factor
from shellpy.fsdt6.constitutive_matrix_in_shell_frame import constitutive_matrix_in_shell_frame
from shellpy.fsdt7_eas.constitutive_matriz_in_material_frame import constitutive_matrix_in_material_frame
from shellpy.fsdt6.strain_vector import linear_strain_vector, nonlinear_strain_vector
from shellpy.numeric_integration.integral_weights import double_integral_weights
import time


@cache_function
def _pre_calculus(shell: Shell, eas_field, n_x, n_y, n_z,
                  integral_method):

    # Get integration points and weights for the double integral over the mid-surface domain
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)
    h = shell.thickness(xi1, xi2)
    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    # Shape of xi1 (discretized domain in terms of xi1 and xi2)
    n_xy = np.shape(xi1)

    # Number of degrees of freedom (dof) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)
    Wxy1 = sqrtG * Wxy

    # Calculate the constitutive tensor C for the thin shell material (cópia para não alterar cache)
    C_material = np.copy(
        constitutive_matrix_in_material_frame(shell.mid_surface_geometry, shell.material, (xi1, xi2, xi3)))
    if C_material.ndim == 2:
        C_material = np.einsum('ij, xyz->ijxyz', C_material, xi3 ** 0)
    kappa_x, kappa_y, kappa_xy = shear_correction_factor(C_material, xi3, Wz, det_shifter_tensor)

    print('kappa_x = ', kappa_x.mean(), 'kappa_y = ', kappa_y.mean(), 'kappa_xy = ', kappa_xy.mean())

    C_material[4, 4] = np.einsum('xyz, xy->xyz', C_material[4, 4], kappa_x)
    C_material[3, 3] = np.einsum('xyz, xy->xyz', C_material[3, 3], kappa_y)
    C_material[3, 4] = np.einsum('xyz, xy->xyz', C_material[3, 4], kappa_xy)
    C_material[4, 3] = np.einsum('xyz, xy->xyz', C_material[4, 3], kappa_xy)

    C = constitutive_matrix_in_shell_frame(shell.mid_surface_geometry, C_material, (xi1, xi2, xi3))

    # Integração do tensor constitutivo na espessura (pesos pré‑computados)
    detWz = det_shifter_tensor * Wz
    C0 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 0 * detWz, optimize=True)
    C1 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 1 * detWz, optimize=True)
    C2 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 2 * detWz, optimize=True)
    C3 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 3 * detWz, optimize=True)
    C4 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 4 * detWz, optimize=True)

    # Deformações lineares [B_L]- Initialize arrays for linear strain components (gamma_lin) and their associated quantities (rho_lin)
    print('Calculating linear strain components...')
    epsilon0_lin = np.zeros((n_dof, 6) + n_xy)
    epsilon1_lin = np.zeros((n_dof, 6) + n_xy)
    epsilon2_lin = np.zeros((n_dof, 6) + n_xy)

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i], epsilon2_lin[i] = linear_strain_vector(shell.mid_surface_geometry,
                                                                                 shell.displacement_expansion,
                                                                                 i, xi1, xi2)

        print(f'Calculating linear components {i + 1} of {n_dof}')

    # Deformações não lineares
    print('Calculating nonlinear strain components...')
    epsilon0_nl = np.zeros((n_dof, n_dof, 6) + n_xy)
    epsilon1_nl = np.zeros((n_dof, n_dof, 6) + n_xy)
    epsilon2_nl = np.zeros((n_dof, n_dof, 6) + n_xy)

    for i in range(n_dof):
        for j in range(i, n_dof):  # E_{ij} = E_{ji}
            epsilon0_nl[i, j], epsilon1_nl[i, j], epsilon2_nl[i, j] = nonlinear_strain_vector(
                shell.mid_surface_geometry,
                shell.displacement_expansion,
                i, j, xi1, xi2)

            epsilon0_nl[j, i], epsilon1_nl[j, i], epsilon2_nl[j, i] = epsilon0_nl[i, j], epsilon1_nl[i, j], epsilon2_nl[
                i, j]

            print(f'Calculating nonlinear components ({i + 1}, {j + 1}) of ({n_dof}, {n_dof})')

    # Campo EAS  [B_α]
    print('Calculating enhanced assumed strain...')
    mu = enhanced_assumed_strain(eas_field, (xi1, xi2, xi3), Wxy1)
    L0_alpha = np.einsum('ijxy, njxy->nixy', C1[:, [2]], mu, optimize=True)
    L1_alpha = np.einsum('ijxy, njxy->nixy', C2[:, [2]], mu, optimize=True)
    L2_alpha = np.einsum('ijxy, njxy->nixy', C3[:, [2]], mu, optimize=True)

    K_alpha_alpha = np.einsum('ijxy, mixy, njxy, xy->mn', C2[np.ix_([2], [2])], mu, mu, Wxy1, optimize=True)
    K_alpha_alpha_inv = np.linalg.inv(K_alpha_alpha)

    # derivada segunda das deformacoes compativeis
    diff_B0 = epsilon0_nl + np.transpose(epsilon0_nl, [1, 0, 2, 3, 4])
    diff_B1 = epsilon1_nl + np.transpose(epsilon1_nl, [1, 0, 2, 3, 4])
    diff_B2 = epsilon2_nl + np.transpose(epsilon2_nl, [1, 0, 2, 3, 4])

    # derivada dos esforcos internos
    diff_L0 = (np.einsum('abxy, imbxy->imaxy', C0, epsilon0_nl, optimize=True) +
               np.einsum('abxy, imbxy->imaxy', C1, epsilon1_nl, optimize=True) +
               np.einsum('abxy, imbxy->imaxy', C2, epsilon2_nl, optimize=True))

    diff_L1 = (np.einsum('abxy, imbxy->imaxy', C1, epsilon0_nl, optimize=True) +
               np.einsum('abxy, imbxy->imaxy', C2, epsilon1_nl, optimize=True) +
               np.einsum('abxy, imbxy->imaxy', C3, epsilon2_nl, optimize=True))

    diff_L2 = (np.einsum('abxy, imbxy->imaxy', C2, epsilon0_nl, optimize=True) +
               np.einsum('abxy, imbxy->imaxy', C3, epsilon1_nl, optimize=True) +
               np.einsum('abxy, imbxy->imaxy', C4, epsilon2_nl, optimize=True))

    diff_K_u_alpha_x_x = (np.einsum('kmaxy, paxy, xy->kmp', diff_B0, L0_alpha, Wxy1, optimize=True) +
                          np.einsum('kmaxy, paxy, xy->kmp', diff_B1, L1_alpha, Wxy1, optimize=True) +
                          np.einsum('kmaxy, paxy, xy->kmp', diff_B2, L2_alpha, Wxy1, optimize=True))



    return Wxy1, C0, C1, C2, C3, C4, epsilon0_lin, epsilon1_lin, epsilon2_lin, epsilon0_nl, epsilon1_nl, epsilon2_nl, mu, L0_alpha, L1_alpha, L2_alpha, K_alpha_alpha_inv, diff_B0, diff_B1, diff_B2, diff_L0, diff_L1, diff_L2, diff_K_u_alpha_x_x

@cache_function
def _pre_calculus2(u, Wxy1, C0, C1, C2, C3, C4, epsilon0_lin, epsilon1_lin, epsilon2_lin, epsilon0_nl, epsilon1_nl, epsilon2_nl, mu, L0_alpha, L1_alpha, L2_alpha):
    #print("pre2")
    # deformacoes compativeis

    t0 = time.perf_counter()

    E0 = epsilon0_lin + np.einsum('ijaxy, j->iaxy', epsilon0_nl, u)
    E1 = epsilon1_lin + np.einsum('ijaxy, j->iaxy', epsilon1_nl, u)
    E2 = epsilon2_lin + np.einsum('ijaxy, j->iaxy', epsilon2_nl, u)

    t1 = time.perf_counter()

    # derivadas das deformacoes compativeis
    B0 = E0 + np.einsum('jiaxy, j->iaxy', epsilon0_nl, u)
    B1 = E1 + np.einsum('jiaxy, j->iaxy', epsilon1_nl, u)
    B2 = E2 + np.einsum('jiaxy, j->iaxy', epsilon2_nl, u)

    t2 = time.perf_counter()

    # esforcos internos compativeis
    L0 = (np.einsum('abxy, ibxy->iaxy', C0, E0) +
          np.einsum('abxy, ibxy->iaxy', C1, E1) +
          np.einsum('abxy, ibxy->iaxy', C2, E2))

    L1 = (np.einsum('abxy, ibxy->iaxy', C1, E0) +
          np.einsum('abxy, ibxy->iaxy', C2, E1) +
          np.einsum('abxy, ibxy->iaxy', C3, E2))

    L2 = (np.einsum('abxy, ibxy->iaxy', C2, E0) +
          np.einsum('abxy, ibxy->iaxy', C3, E1) +
          np.einsum('abxy, ibxy->iaxy', C4, E2))

    t3 = time.perf_counter()

    # matrizes de rigidez
    K_alpha_u = np.einsum('maxy, naxy, xy->mn', mu, L1[:, [2]], Wxy1)

    t4 = time.perf_counter()

    K_u_u = (np.einsum('maxy, naxy, xy->mn', B0, L0, Wxy1, optimize=True) +
             np.einsum('maxy, naxy, xy->mn', B1, L1, Wxy1, optimize=True) +
             np.einsum('maxy, naxy, xy->mn', B2, L2, Wxy1, optimize=True))

    t5 = time.perf_counter()

    K_u_alpha = (np.einsum('maxy, naxy, xy->mn', B0, L0_alpha, Wxy1, optimize=True) +
                 np.einsum('maxy, naxy, xy->mn', B1, L1_alpha, Wxy1, optimize=True) +
                 np.einsum('maxy, naxy, xy->mn', B2, L2_alpha, Wxy1, optimize=True))

    t6 = time.perf_counter()

    print(f"_pre E: {t1 - t0:.6f} s")
    print(f"_pre B: {t2 - t1:.6f} s")
    print(f"_pre L: {t3 - t2:.6f} s")
    print(f"_pre K_alpha_u: {t4 - t3:.6f} s")
    print(f"_pre K_u_u: {t5 - t4:.6f} s")
    print(f"_pre K_u_alpha: {t6 - t5:.6f} s")
    print(f"tempo total pre: {t6 - t0:.6f} s")

    return E0, E1, E2, B0, B1, B2, L0, L1, L2, K_u_u, K_u_alpha, K_alpha_u
