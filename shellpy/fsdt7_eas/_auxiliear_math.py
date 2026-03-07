import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def calc_nl_numba(epsilon_nl, u):
    # epsilon_nl: (dof, dof, 6, x, y)
    # u: (dof,)
    n_dof, _, n_stress, n_x, n_y = epsilon_nl.shape
    # Inicializa o output (iaxy)
    res = np.zeros((n_dof, n_stress, n_x, n_y))

    # prange distribui os DOFs entre os núcleos do CPU
    for i in prange(n_dof):
        for j in range(n_dof):
            u_val = u[j]
            if u_val == 0: continue  # Otimização para vetores esparsos
            for a in range(n_stress):
                for x in range(n_x):
                    for y in range(n_y):
                        res[i, a, x, y] += epsilon_nl[i, j, a, x, y] * u_val
    return res


@njit(parallel=True, fastmath=True)
def calc_Kuu_numba(B, Lw):
    # B e Lw: (dof, 6, x, y)
    n_dof = B.shape[0]
    n_stress = B.shape[1]
    n_x = B.shape[2]
    n_y = B.shape[3]

    K = np.zeros((n_dof, n_dof))

    for m in prange(n_dof):
        for n in range(n_dof):
            sum_val = 0.0
            for a in range(n_stress):
                for x in range(n_x):
                    for y in range(n_y):
                        sum_val += B[m, a, x, y] * Lw[n, a, x, y]
            K[m, n] = sum_val
    return K


@njit(parallel=True, fastmath=True)
def compute_diff_K_u_u_numba(
        diff_B0, diff_B1, diff_B2,
        Lu0w, Lu1w, Lu2w,
        B0, B1, B2,
        dLu0w, dLu1w, dLu2w
):
    # Dimensões: k, m (DOFs), a (stress components 6), x, y (grid 20x20)
    n_k, n_m, n_a, n_x, n_y = diff_B0.shape

    # Inicializa a matriz de saída km
    diff_K = np.zeros((n_k, n_m))

    # prange distribui as linhas 'k' entre os núcleos do CPU
    for k in prange(n_k):
        for m in range(n_m):
            acc = 0.0
            for a in range(n_a):
                for x in range(n_x):
                    for y in range(n_y):
                        # Bloco 1: Contrações kmaxy * axy
                        term1 = (diff_B0[k, m, a, x, y] * Lu0w[a, x, y] +
                                 diff_B1[k, m, a, x, y] * Lu1w[a, x, y] +
                                 diff_B2[k, m, a, x, y] * Lu2w[a, x, y])

                        # Bloco 2: Contrações kaxy * maxy
                        term2 = (B0[k, a, x, y] * dLu0w[m, a, x, y] +
                                 B1[k, a, x, y] * dLu1w[m, a, x, y] +
                                 B2[k, a, x, y] * dLu2w[m, a, x, y])

                        acc += term1 + term2

            diff_K[k, m] = acc

    return diff_K

