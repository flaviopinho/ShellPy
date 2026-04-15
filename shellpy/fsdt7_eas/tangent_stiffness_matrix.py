import time
import numpy as np
from shellpy import Shell
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
from shellpy.fsdt7_eas._compute_constant_shell_matrices import compute_constant_shell_matrices
from shellpy.fsdt7_eas._compute_displacement_dependent_matrices import compute_displacement_dependent_matrices


def tangent_stiffness_matrix(u, shell: Shell, eas_field, n_x=20, n_y=20, n_z=10,
                             integral_method=gauss_weights_simple_integral):
    """
    Calcula a Matriz de Rigidez Tangente (Jacobiana) exata para o método de Newton-Raphson.
    Totalmente otimizada via BLAS (Matrix-Vector e Matrix-Matrix products).
    """
    t_total_start = time.perf_counter()

    # --------------------------------------------------------------------------------
    # 1. RECUPERAÇÃO DE DADOS (ESTÁTICOS E DEPENDENTES DE U)
    # --------------------------------------------------------------------------------
    # Recupera os dados pré-calculados (Constantes)
    (Wxy1_flat, C0_T, C1_T, C2_T, C3_T, C4_T,
     eps0_lin_flat, eps1_lin_flat, eps2_lin_flat,
     eps0_nl_matrix, eps1_nl_matrix, eps2_nl_matrix,
     mu_flat, L0_alpha_W_flat, L1_alpha_W_flat, L2_alpha_W_flat,
     K_alpha_alpha_inv, diff_B0_flat, diff_B1_flat, diff_B2_flat,
     diff_L0_flat, diff_L1_flat, diff_L2_flat, diff_K_u_alpha_x_x_raw, n_xy) = compute_constant_shell_matrices(
        shell, eas_field, n_x, n_y, n_z, integral_method
    )

    # Recupera variáveis de estado (Hot Loop)
    (E0, E1, E2, B0, B1, B2, L0, L1, L2,
     K_u_u, K_u_alpha, K_alpha_u) = compute_displacement_dependent_matrices(
        u, Wxy1_flat, C0_T, C1_T, C2_T, C3_T, C4_T,
        eps0_lin_flat, eps1_lin_flat, eps2_lin_flat,
        eps0_nl_matrix, eps1_nl_matrix, eps2_nl_matrix,
        mu_flat, L0_alpha_W_flat, L1_alpha_W_flat, L2_alpha_W_flat, n_xy
    )
    t_hotloop_end = time.perf_counter()

    n_dof = u.shape[0]
    n_points = Wxy1_flat.size

    # --------------------------------------------------------------------------------
    # 2. MATRIZ SECANTE CONDENSADA (EAS)
    # --------------------------------------------------------------------------------
    # Condensação estática dos parâmetros Alpha no nível da tangente
    K_secant_condensed = K_u_u - K_u_alpha @ K_alpha_alpha_inv @ K_alpha_u
    t_secante_end = time.perf_counter()

    # --------------------------------------------------------------------------------
    # 3. RIGIDEZ GEOMÉTRICA (K_geo + K_mat_tangent) - OTIMIZAÇÃO BLAS
    # --------------------------------------------------------------------------------
    # Parte A: Matriz de Tensão Inicial (Initial Stress Matrix)
    # Primeiro, calculamos o esforço real integrado para o ponto atual
    # (u @ L) reduz (dof) @ (dof, 6, k) -> (6, k)
    s0 = ((u @ L0.reshape(n_dof, -1)).reshape(6, n_points) * Wxy1_flat).reshape(-1)
    s1 = ((u @ L1.reshape(n_dof, -1)).reshape(6, n_points) * Wxy1_flat).reshape(-1)
    s2 = ((u @ L2.reshape(n_dof, -1)).reshape(6, n_points) * Wxy1_flat).reshape(-1)

    # Contração via BLAS: (dof*dof, 6*k) @ (6*k) -> (dof, dof)
    K_geo = (diff_B0_flat.reshape(n_dof ** 2, -1) @ s0 +
             diff_B1_flat.reshape(n_dof ** 2, -1) @ s1 +
             diff_B2_flat.reshape(n_dof ** 2, -1) @ s2).reshape(n_dof, n_dof)

    # Parte B: Rigidez Material Tangente (Variação dos Esforços dL)
    # duL é a variação do esforço L com relação a u: (dof, 6, k)
    duL0 = (u @ diff_L0_flat.reshape(n_dof, -1)).reshape(n_dof, 6, n_points)
    duL1 = (u @ diff_L1_flat.reshape(n_dof, -1)).reshape(n_dof, 6, n_points)
    duL2 = (u @ diff_L2_flat.reshape(n_dof, -1)).reshape(n_dof, 6, n_points)

    # Integração: B^T @ (duL * W) -> (dof, 6*k) @ (6*k, dof)
    K_mat_tangent = (B0.reshape(n_dof, -1) @ (duL0 * Wxy1_flat).reshape(n_dof, -1).T +
                     B1.reshape(n_dof, -1) @ (duL1 * Wxy1_flat).reshape(n_dof, -1).T +
                     B2.reshape(n_dof, -1) @ (duL2 * Wxy1_flat).reshape(n_dof, -1).T)

    diff_K_u_u = K_geo + K_mat_tangent
    t_geometrica_end = time.perf_counter()

    # --------------------------------------------------------------------------------
    # 4. TERMOS TANGENTES DO CAMPO EAS (CORREÇÃO)
    # --------------------------------------------------------------------------------
    # Correção do acoplamento entre deslocamentos compatíveis e enriquecidos
    aux_alpha = K_alpha_alpha_inv @ (K_alpha_u @ u)
    dk_u_a_x = np.dot(diff_K_u_alpha_x_x_raw, aux_alpha)

    # Variação do campo EAS (diff_K_alpha_u) via BLAS
    dL1_shear_flat = diff_L1_flat[:, :, 2, :].reshape(n_dof, -1)  # dof, dof*k
    dL1_u = (u @ dL1_shear_flat).reshape(n_dof, n_points)  # dof, k

    mu_W = mu_flat * Wxy1_flat  # n_alpha, k
    diff_K_alpha_u = mu_W @ dL1_u.T

    t_eas_end = time.perf_counter()

    # --------------------------------------------------------------------------------
    # 5. MONTAGEM FINAL DA JACOBIANA
    # --------------------------------------------------------------------------------
    Jacobian = K_secant_condensed + diff_K_u_u - dk_u_a_x - (K_u_alpha @ K_alpha_alpha_inv @ diff_K_alpha_u)
    t_total_end = time.perf_counter()

    # --------------------------------------------------------------------------------
    # 6. RELATÓRIO DE PERFORMANCE
    # --------------------------------------------------------------------------------
    print(f"\n--- [TANGENT STIFFNESS MATRIX DIAGNOSTICS] ---")
    print(f"Estado/HotLoop     : {t_hotloop_end - t_total_start:.6f} s")
    print(f"Secante Condensada : {t_secante_end - t_hotloop_end:.6f} s")
    print(f"Rigidez Geométrica : {t_geometrica_end - t_secante_end:.6f} s (Otimizada BLAS)")
    print(f"Correção EAS       : {t_eas_end - t_geometrica_end:.6f} s")
    print(f"TOTAL JACOBIANA    : {t_total_end - t_total_start:.6f} s")
    print(f"----------------------------------------------\n")

    return Jacobian