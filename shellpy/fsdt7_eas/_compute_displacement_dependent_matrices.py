import time
from functools import wraps

import numpy as np


def cache_last_u(func):
    """
    Cache otimizado para solvers não lineares (como Newton-Raphson).
    Armazena apenas a última avaliação baseada no vetor de deslocamento `u`.
    Evita que o loop principal recalcule matrizes pesadas se `u` não mudou,
    sem causar vazamento de memória (memory leak) associado a caches padrão.
    """
    last_u = None
    last_result = None

    @wraps(func)
    def wrapper(u, *args, **kwargs):
        nonlocal last_u, last_result

        # Se u for idêntico ao anterior, retorna o valor em cache (curto-circuito)
        if last_u is not None and np.array_equal(u, last_u):
            return last_result

        # Caso contrário, calcula tudo novamente
        result = func(u, *args, **kwargs)

        # Atualiza o cache
        last_u = u.copy()
        last_result = result

        return result

    return wrapper


@cache_last_u
def compute_displacement_dependent_matrices(
        u, Wxy1_flat,
        C0_T, C1_T, C2_T, C3_T, C4_T,
        eps0_lin_flat, eps1_lin_flat, eps2_lin_flat,
        eps0_nl_matrix, eps1_nl_matrix, eps2_nl_matrix,
        mu_flat, L0_alpha_W_flat, L1_alpha_W_flat, L2_alpha_W_flat, n_xy):
    """
    Calcula as matrizes e vetores que dependem do estado atual de deslocamento (u).
    Esta função é o 'hot loop' do método de Newton-Raphson, sendo executada em cada iteração.
    """
    t_inicio = time.perf_counter()
    u = np.ascontiguousarray(u)

    # --------------------------------------------------------------------------------
    # 1. CINEMÁTICA: DEFORMAÇÕES REAIS (E) E OPERADOR TANGENTE (B)
    # --------------------------------------------------------------------------------
    t0 = time.perf_counter()
    n_dof, n_stress, n_points = eps0_lin_flat.shape

    # Cálculo da contribuição não linear via contração (Otimizado via BLAS @)
    # Transforma o tensor de deformação não linear em um vetor de estado baseado em u
    E0_nl = (eps0_nl_matrix @ u).reshape(n_dof, n_stress, n_points)
    E1_nl = (eps1_nl_matrix @ u).reshape(n_dof, n_stress, n_points)
    E2_nl = (eps2_nl_matrix @ u).reshape(n_dof, n_stress, n_points)

    # Deformações compatíveis totais (E = E_lin + E_nl)
    E0 = eps0_lin_flat + E0_nl
    E1 = eps1_lin_flat + E1_nl
    E2 = eps2_lin_flat + E2_nl

    # Operadores cinemáticos tangentes (B = B_lin + 2 * E_nl)
    B0 = eps0_lin_flat + 2.0 * E0_nl
    B1 = eps1_lin_flat + 2.0 * E1_nl
    B2 = eps2_lin_flat + 2.0 * E2_nl

    t_cinematica = time.perf_counter() - t0

    # --------------------------------------------------------------------------------
    # 2. ESFORÇOS INTERNOS E RESULTANTES DE TENSÃO (L)
    # --------------------------------------------------------------------------------
    t1 = time.perf_counter()

    # Cálculo dos esforços resultantes (L0, L1, L2) através da espessura integrada
    # Utiliza einsum para contração entre os tensores constitutivos (C) e as deformações (E)
    L0 = np.einsum('kab, dbk -> dak', C0_T, E0) + \
         np.einsum('kab, dbk -> dak', C1_T, E1) + \
         np.einsum('kab, dbk -> dak', C2_T, E2)

    L1 = np.einsum('kab, dbk -> dak', C1_T, E0) + \
         np.einsum('kab, dbk -> dak', C2_T, E1) + \
         np.einsum('kab, dbk -> dak', C3_T, E2)

    L2 = np.einsum('kab, dbk -> dak', C2_T, E0) + \
         np.einsum('kab, dbk -> dak', C3_T, E1) + \
         np.einsum('kab, dbk -> dak', C4_T, E2)

    t_esforcos_L = time.perf_counter() - t1

    # --------------------------------------------------------------------------------
    # 3. MATRIZES DE RIGIDEZ ELÁSTICA (K)
    # --------------------------------------------------------------------------------
    t2 = time.perf_counter()

    # Preparações e Achatamento (Reshaping) para multiplicação de matrizes 2D
    B0_flat = B0.reshape(n_dof, -1)
    B1_flat = B1.reshape(n_dof, -1)
    B2_flat = B2.reshape(n_dof, -1)

    # Aplicação dos pesos de integração (W) e achatamento dos esforços
    L0_W_flat = (L0 * Wxy1_flat).reshape(n_dof, -1)
    L1_W_flat = (L1 * Wxy1_flat).reshape(n_dof, -1)
    L2_W_flat = (L2 * Wxy1_flat).reshape(n_dof, -1)

    # Montagem da matriz de rigidez U-U (Deslocamento-Deslocamento)
    K_u_u = B0_flat @ L0_W_flat.T + B1_flat @ L1_W_flat.T + B2_flat @ L2_W_flat.T

    # Matriz de acoplamento U-Alpha (Deslocamento-EAS)
    K_u_alpha = B0_flat @ L0_alpha_W_flat.T + B1_flat @ L1_alpha_W_flat.T + B2_flat @ L2_alpha_W_flat.T

    # Componente da matriz de rigidez Alpha-U (EAS-Deslocamento)
    # Focada na componente de cisalhamento/estiramento específica (slice 2:3)
    L1_slice_W_flat = (L1[:, 2:3, :] * Wxy1_flat).reshape(n_dof, -1)
    K_alpha_u = mu_flat @ L1_slice_W_flat.T

    t_rigidez_K = time.perf_counter() - t2

    # --------------------------------------------------------------------------------
    # 4. LOG DE PERFORMANCE E RETORNO
    # --------------------------------------------------------------------------------
    t_total = time.perf_counter() - t_inicio

    print(f"\n" + "-" * 40)
    print(f"PERFORMANCE LOOP (Iteração)")
    print(f"-" * 40)
    print(f"1. Cinemática (E & B)      : {t_cinematica:.6f} s")
    print(f"2. Esforços Internos (L)   : {t_esforcos_L:.6f} s")
    print(f"3. Matrizes de Rigidez (K) : {t_rigidez_K:.6f} s")
    print(f"TOTAL DO CICLO             : {t_total:.6f} s")
    print("-" * 40 + "\n")

    return E0, E1, E2, B0, B1, B2, L0, L1, L2, K_u_u, K_u_alpha, K_alpha_u
