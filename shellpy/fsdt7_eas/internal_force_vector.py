import time
import numpy as np
from shellpy import Shell
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
from shellpy.fsdt7_eas._compute_constant_shell_matrices import compute_constant_shell_matrices
from shellpy.fsdt7_eas._compute_displacement_dependent_matrices import compute_displacement_dependent_matrices


def internal_force_vector(u, shell: Shell, eas_field, n_x=20, n_y=20, n_z=10,
                          integral_method=gauss_weights_simple_integral):
    """
    Calcula o vetor de forças internas da estrutura para um dado vetor de deslocamentos 'u'.
    Utiliza a formulação Enhanced Assumed Strain (EAS) com condensação estática.
    """
    t0 = time.perf_counter()

    # --------------------------------------------------------------------------------
    # 1. PRÉ-CÁLCULO (RECUPERAÇÃO DO CACHE)
    # --------------------------------------------------------------------------------
    (Wxy1_flat, C0_T, C1_T, C2_T, C3_T, C4_T,
     eps0_lin_flat, eps1_lin_flat, eps2_lin_flat,
     eps0_nl_flat, eps1_nl_flat, eps2_nl_flat,
     mu_flat, L0_alpha_W_flat, L1_alpha_W_flat, L2_alpha_W_flat,
     K_alpha_alpha_inv, diff_B0_flat, diff_B1_flat, diff_B2_flat,
     diff_L0_flat, diff_L1_flat, diff_L2_flat, diff_K_u_alpha_x_x, n_xy) = compute_constant_shell_matrices(
        shell, eas_field, n_x, n_y, n_z, integral_method
    )

    t1 = time.perf_counter()

    # --------------------------------------------------------------------------------
    # 2. VARIÁVEIS DE ESTADO ("HOT LOOP") - ATUALIZADO
    # --------------------------------------------------------------------------------
    # Ajustado para receber os retornos achatados (E0...L2) e as matrizes elásticas.
    # Note que não usamos E, B e L aqui, mas precisamos recebê-los para chegar no K_u_u.
    (E0, E1, E2, B0, B1, B2, L0, L1, L2,
     K_u_u, K_u_alpha, K_alpha_u) = compute_displacement_dependent_matrices(
        u, Wxy1_flat, C0_T, C1_T, C2_T, C3_T, C4_T,
        eps0_lin_flat, eps1_lin_flat, eps2_lin_flat,
        eps0_nl_flat, eps1_nl_flat, eps2_nl_flat,
        mu_flat, L0_alpha_W_flat, L1_alpha_W_flat, L2_alpha_W_flat, n_xy
    )

    t2 = time.perf_counter()

    # --------------------------------------------------------------------------------
    # 3. CONDENSAÇÃO ESTÁTICA DO CAMPO EAS
    # --------------------------------------------------------------------------------
    # K_secant = K_uu - K_ua @ K_aa^-1 @ K_au
    # Operação puramente matricial (BLAS), extremamente rápida.
    K_secant_condensed = K_u_u - K_u_alpha @ K_alpha_alpha_inv @ K_alpha_u

    t3 = time.perf_counter()

    # --------------------------------------------------------------------------------
    # 4. VETOR DE FORÇAS INTERNAS
    # --------------------------------------------------------------------------------
    F_int = K_secant_condensed @ u

    t4 = time.perf_counter()

    # --------------------------------------------------------------------------------
    # 5. DIAGNÓSTICO DE PERFORMANCE
    # --------------------------------------------------------------------------------
    print(f"--- Performance Breakdown: Forças Internas ---")
    print(f"Recuperação Cache (Pre) : {t1 - t0:.6f} s")
    print(f"Variáveis de Estado     : {t2 - t1:.6f} s")
    print(f"Condensação EAS (K_sec) : {t3 - t2:.6f} s")
    print(f"Produto K_sec @ u (F_int): {t4 - t3:.6f} s")
    print(f"Tempo Total do Ciclo    : {t4 - t0:.6f} s")
    print(f"----------------------------------------------")

    return F_int