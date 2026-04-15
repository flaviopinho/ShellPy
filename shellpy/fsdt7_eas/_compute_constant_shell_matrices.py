import numpy as np
from shellpy import Shell, cache_function
from shellpy.fsdt7_eas.enhanced_assumed_strain import enhanced_assumed_strain_full
from shellpy.fsdt6.shear_correction_factor import shear_correction_factor
from shellpy.fsdt6.constitutive_matrix_in_shell_frame import constitutive_matrix_in_shell_frame
from shellpy.fsdt7_eas.constitutive_matriz_in_material_frame import constitutive_matrix_in_material_frame
from shellpy.fsdt6.strain_vector import linear_strain_vector, nonlinear_strain_vector
from shellpy.numeric_integration.integral_weights import double_integral_weights


@cache_function
def compute_constant_shell_matrices(shell, eas_field, n_x, n_y, n_z, integral_method):
    """
    Realiza o cálculo de todas as matrizes e tensores da casca que independem do deslocamento (u).
    """
    # --------------------------------------------------------------------------------
    # 1. GEOMETRIA E INTEGRAÇÃO NUMÉRICA
    # --------------------------------------------------------------------------------
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)
    h = shell.thickness(xi1, xi2)
    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    n_xy = np.shape(xi1)
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)
    Wxy1 = sqrtG * Wxy

    # --------------------------------------------------------------------------------
    # 2. PROPRIEDADES DO MATERIAL E TENSOR CONSTITUTIVO
    # --------------------------------------------------------------------------------
    C_material = np.copy(
        constitutive_matrix_in_material_frame(shell.mid_surface_geometry, shell.material, (xi1, xi2, xi3)))
    if C_material.ndim == 2:
        C_material = np.einsum('ij, xyz->ijxyz', C_material, xi3 ** 0)

    kappa_x, kappa_y, kappa_xy = shear_correction_factor(C_material, xi3, Wz, det_shifter_tensor)
    C_material[4, 4] = np.einsum('xyz, xy->xyz', C_material[4, 4], kappa_x)
    C_material[3, 3] = np.einsum('xyz, xy->xyz', C_material[3, 3], kappa_y)
    C_material[3, 4] = np.einsum('xyz, xy->xyz', C_material[3, 4], kappa_xy)
    C_material[4, 3] = np.einsum('xyz, xy->xyz', C_material[4, 3], kappa_xy)

    C = constitutive_matrix_in_shell_frame(shell.mid_surface_geometry, C_material, (xi1, xi2, xi3))

    detWz = det_shifter_tensor * Wz
    C0 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 0 * detWz, optimize=True)
    C1 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 1 * detWz, optimize=True)
    C2 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 2 * detWz, optimize=True)
    C3 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 3 * detWz, optimize=True)
    C4 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 4 * detWz, optimize=True)

    # --------------------------------------------------------------------------------
    # 3. KINEMÁTICA: DEFORMAÇÕES LINEARES E NÃO LINEARES (Malha Cheia)
    # --------------------------------------------------------------------------------
    print('Calculando componentes da deformação linear...')
    epsilon0_lin = np.zeros((n_dof, 6) + n_xy)
    epsilon1_lin = np.zeros((n_dof, 6) + n_xy)
    epsilon2_lin = np.zeros((n_dof, 6) + n_xy)

    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i], epsilon2_lin[i] = linear_strain_vector(
            shell.mid_surface_geometry, shell.displacement_expansion, i, xi1, xi2)

    print('Calculando componentes da deformação não linear...')
    epsilon0_nl = np.zeros((n_dof, n_dof, 6) + n_xy)
    epsilon1_nl = np.zeros((n_dof, n_dof, 6) + n_xy)
    epsilon2_nl = np.zeros((n_dof, n_dof, 6) + n_xy)

    for i in range(n_dof):
        for j in range(i, n_dof):
            res = nonlinear_strain_vector(shell.mid_surface_geometry, shell.displacement_expansion, i, j, xi1, xi2)
            epsilon0_nl[i, j], epsilon1_nl[i, j], epsilon2_nl[i, j] = res
            epsilon0_nl[j, i], epsilon1_nl[j, i], epsilon2_nl[j, i] = res

    # ================================================================================
    # Transforma matrizes 2D espaciais em 1D apenas com os pontos ativos.
    # ================================================================================

    def prune(tensor_xy):
        """Transforma matrizes 2D espaciais em 1D apenas com os pontos ativos."""
        shape_base = tensor_xy.shape[:-2]
        return np.ascontiguousarray(tensor_xy.reshape(*shape_base, -1))

    Wxy1_flat = prune(Wxy1)

    # Constitutivas Reduzidas (6, 6, n_ativos)
    C0_p, C1_p, C2_p, C3_p, C4_p = prune(C0), prune(C1), prune(C2), prune(C3), prune(C4)

    # Deformações Lineares Reduzidas (n_dof, 6, n_ativos)
    e0_l_p, e1_l_p, e2_l_p = prune(epsilon0_lin), prune(epsilon1_lin), prune(epsilon2_lin)

    # Deformações Não-Lineares Reduzidas (n_dof, n_dof, 6, n_ativos)
    e0_nl_p, e1_nl_p, e2_nl_p = prune(epsilon0_nl), prune(epsilon1_nl), prune(epsilon2_nl)

    # --------------------------------------------------------------------------------
    # 4. FORMULAÇÃO EAS
    # --------------------------------------------------------------------------------
    print('Calculando campo EAS...')

    mu = enhanced_assumed_strain_full(eas_field, (xi1, xi2, xi3), Wxy1, detWz)

    L0_alpha = np.einsum('ijxy, njxy->nixy', C1[:, [2]], mu, optimize=True)
    L1_alpha = np.einsum('ijxy, njxy->nixy', C2[:, [2]], mu, optimize=True)
    L2_alpha = np.einsum('ijxy, njxy->nixy', C3[:, [2]], mu, optimize=True)

    # Trocamos para 's' para evitar colisões
    mu_p = prune(mu)
    K_alpha_alpha = np.einsum('ijs, mis, njs, s->mn', C2_p[np.ix_([2], [2])], mu_p, mu_p, Wxy1_flat, optimize=True)
    K_alpha_alpha_inv = np.linalg.inv(K_alpha_alpha)

    L0_alpha_p = prune(L0_alpha)
    L1_alpha_p = prune(L1_alpha)
    L2_alpha_p = prune(L2_alpha)

    # --------------------------------------------------------------------------------
    # 5. DERIVADAS PARA A MATRIZ TANGENTE
    # --------------------------------------------------------------------------------

    print('Calculando Tensores Tangentes...')
    diff_B0 = e0_nl_p + np.transpose(e0_nl_p, [1, 0, 2, 3])
    diff_B1 = e1_nl_p + np.transpose(e1_nl_p, [1, 0, 2, 3])
    diff_B2 = e2_nl_p + np.transpose(e2_nl_p, [1, 0, 2, 3])

    # Usando o índice 's' para a dimensão espacial
    diff_L0 = (np.einsum('abs, imbs->imas', C0_p, e0_nl_p, optimize=True) +
               np.einsum('abs, imbs->imas', C1_p, e1_nl_p, optimize=True) +
               np.einsum('abs, imbs->imas', C2_p, e2_nl_p, optimize=True))

    diff_L1 = (np.einsum('abs, imbs->imas', C1_p, e0_nl_p, optimize=True) +
               np.einsum('abs, imbs->imas', C2_p, e1_nl_p, optimize=True) +
               np.einsum('abs, imbs->imas', C3_p, e2_nl_p, optimize=True))

    diff_L2 = (np.einsum('abs, imbs->imas', C2_p, e0_nl_p, optimize=True) +
               np.einsum('abs, imbs->imas', C3_p, e1_nl_p, optimize=True) +
               np.einsum('abs, imbs->imas', C4_p, e2_nl_p, optimize=True))

    # Sem colisão: 'k' e 'm' são DOFs, 'p' é o EAS, 's' é o espaço.
    diff_K_u_alpha_x_x = (np.einsum('kmas, pas, s->kmp', diff_B0, L0_alpha_p, Wxy1_flat, optimize=True) +
                          np.einsum('kmas, pas, s->kmp', diff_B1, L1_alpha_p, Wxy1_flat, optimize=True) +
                          np.einsum('kmas, pas, s->kmp', diff_B2, L2_alpha_p, Wxy1_flat, optimize=True))

    # --------------------------------------------------------------------------------
    # 6. FLATTENING FINAL PARA OTIMIZAÇÃO BLAS
    # --------------------------------------------------------------------------------
    # Matrizes Constitutivas
    C0_T = np.ascontiguousarray(C0_p.transpose(2, 0, 1))
    C1_T = np.ascontiguousarray(C1_p.transpose(2, 0, 1))
    C2_T = np.ascontiguousarray(C2_p.transpose(2, 0, 1))
    C3_T = np.ascontiguousarray(C3_p.transpose(2, 0, 1))
    C4_T = np.ascontiguousarray(C4_p.transpose(2, 0, 1))

    # Deformações
    eps0_lin_flat = np.ascontiguousarray(e0_l_p)
    eps1_lin_flat = np.ascontiguousarray(e1_l_p)
    eps2_lin_flat = np.ascontiguousarray(e2_l_p)

    eps0_nl_matrix = np.ascontiguousarray(e0_nl_p.transpose(0, 2, 3, 1).reshape(-1, n_dof))
    eps1_nl_matrix = np.ascontiguousarray(e1_nl_p.transpose(0, 2, 3, 1).reshape(-1, n_dof))
    eps2_nl_matrix = np.ascontiguousarray(e2_nl_p.transpose(0, 2, 3, 1).reshape(-1, n_dof))

    # Derivadas Tangente
    diff_B0_flat = np.ascontiguousarray(diff_B0)
    diff_B1_flat = np.ascontiguousarray(diff_B1)
    diff_B2_flat = np.ascontiguousarray(diff_B2)

    diff_L0_flat = np.ascontiguousarray(diff_L0)
    diff_L1_flat = np.ascontiguousarray(diff_L1)
    diff_L2_flat = np.ascontiguousarray(diff_L2)

    # EAS Flattening (Multiplicando pelo Wxy1_flat esparso)
    n_alpha = L0_alpha.shape[0]
    L0_alpha_W_flat = np.ascontiguousarray((L0_alpha_p * Wxy1_flat).reshape(n_alpha, -1))
    L1_alpha_W_flat = np.ascontiguousarray((L1_alpha_p * Wxy1_flat).reshape(n_alpha, -1))
    L2_alpha_W_flat = np.ascontiguousarray((L2_alpha_p * Wxy1_flat).reshape(n_alpha, -1))
    mu_flat = np.ascontiguousarray(mu_p.reshape(n_alpha, -1))

    def log_mem_mb(name, tensor, factor=1):
        """
        Calcula o tamanho em MB de um tensor.
        O 'factor' multiplica pelo número de tensores similares (ex: C0 a C4 = 5).
        """
        mb = (tensor.nbytes * factor) / (1024 ** 2)
        print(f"{name:<35} | {mb:>12.4f} MB")
        return tensor.nbytes * factor

    total_b = 0
    # Agrupando tensores similares usando o 'factor' para estimar o peso total do bloco
    total_b += log_mem_mb("Matrizes NL (eps0,1,2_nl_matrix)", eps0_nl_matrix, 3)
    total_b += log_mem_mb("Tensores Tangente (diff_L_flat)", diff_L0_flat, 3)
    total_b += log_mem_mb("Tensores Cinemática (diff_B_flat)", diff_B0_flat, 3)
    total_b += log_mem_mb("Matrizes Constitutivas (C0-C4_T)", C0_T, 5)
    total_b += log_mem_mb("Componentes Lineares (eps_lin_flat)", eps0_lin_flat, 3)

    # Se quiser rastrear o EAS também, descomente as linhas abaixo:
    total_b += log_mem_mb("Matrizes EAS (L_alpha_W_flat)", L0_alpha_W_flat, 3)
    total_b += log_mem_mb("Acoplamento EAS (diff_K_u_alpha)", diff_K_u_alpha_x_x, 1)

    total_mb = total_b / (1024 ** 2)
    print("-" * 70)
    print(f"{'TOTAL ACUMULADO EM RAM':<35} | {total_mb:>12.4f} MB")
    print("=" * 70 + "\n")

    return (Wxy1_flat, C0_T, C1_T, C2_T, C3_T, C4_T,
            eps0_lin_flat, eps1_lin_flat, eps2_lin_flat,
            eps0_nl_matrix, eps1_nl_matrix, eps2_nl_matrix,
            mu_flat, L0_alpha_W_flat, L1_alpha_W_flat, L2_alpha_W_flat,
            K_alpha_alpha_inv, diff_B0_flat, diff_B1_flat, diff_B2_flat,
            diff_L0_flat, diff_L1_flat, diff_L2_flat, diff_K_u_alpha_x_x, n_xy)