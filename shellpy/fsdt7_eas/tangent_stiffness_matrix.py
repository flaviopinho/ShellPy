import numpy as np
from shellpy import Shell
from shellpy.fsdt7_eas._auxiliear_math import compute_diff_K_u_u_numba
from shellpy.fsdt7_eas._pre_calculus import _pre_calculus, _pre_calculus2
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
import time


def tangent_stiffness_matrix(u, shell: Shell, eas_field, n_x=20, n_y=20, n_z=10,
                             integral_method=gauss_weights_simple_integral):
    # pre calculo - tudo ja armazenado
    t0 = time.perf_counter()
    Wxy1, C0, C1, C2, C3, C4, epsilon0_lin, epsilon1_lin, epsilon2_lin, epsilon0_nl, epsilon1_nl, epsilon2_nl, mu, L0_alpha, L1_alpha, L2_alpha, K_alpha_alpha_inv, diff_B0, diff_B1, diff_B2, diff_L0, diff_L1, diff_L2, diff_K_u_alpha_x_x = _pre_calculus(
        shell, eas_field, n_x, n_y, n_z, integral_method)
    t1 = time.perf_counter()
    E0, E1, E2, B0, B1, B2, L0, L1, L2, K_u_u, K_u_alpha, K_alpha_u = _pre_calculus2(
        u,
        Wxy1,
        C0, C1, C2, C3, C4,
        epsilon0_lin, epsilon1_lin, epsilon2_lin,
        epsilon0_nl, epsilon1_nl, epsilon2_nl,
        mu, L0_alpha, L1_alpha, L2_alpha)
    t2 = time.perf_counter()

    K = K_u_u - K_u_alpha @ K_alpha_alpha_inv @ K_alpha_u

    t3 = time.perf_counter()

    # contrações com u primeiro
    Lu0w = np.einsum('iaxy,i->axy', L0, u, optimize=True) * Wxy1
    Lu1w = np.einsum('iaxy,i->axy', L1, u, optimize=True) * Wxy1
    Lu2w = np.einsum('iaxy,i->axy', L2, u, optimize=True) * Wxy1

    dLu0w = np.einsum('imaxy,i->maxy', diff_L0, u, optimize=True) * Wxy1
    dLu1w = np.einsum('imaxy,i->maxy', diff_L1, u, optimize=True) * Wxy1
    dLu2w = np.einsum('imaxy,i->maxy', diff_L2, u, optimize=True) * Wxy1

    """
    diff_K_u_u = (
            np.einsum('kmaxy,axy->km', diff_B0, Lu0w, optimize=True) +
            np.einsum('kmaxy,axy->km', diff_B1, Lu1w, optimize=True) +
            np.einsum('kmaxy,axy->km', diff_B2, Lu2w, optimize=True) +
            np.einsum('kaxy,maxy->km', B0, dLu0w, optimize=True) +
            np.einsum('kaxy,maxy->km', B1, dLu1w, optimize=True) +
            np.einsum('kaxy,maxy->km', B2, dLu2w, optimize=True)
    )
    """
    diff_K_u_u = compute_diff_K_u_u_numba(
        diff_B0, diff_B1, diff_B2,
        Lu0w, Lu1w, Lu2w,
        B0, B1, B2,
        dLu0w, dLu1w, dLu2w
    )

    t4 = time.perf_counter()

    aux = K_alpha_alpha_inv @ K_alpha_u @ u

    t5 = time.perf_counter()

    diff_K_u_alpha_x_x = np.einsum('kmp, p->km', diff_K_u_alpha_x_x, aux, optimize=True)

    t6 = time.perf_counter()

    diff_K_alpha_u = np.einsum('kaxy, imaxy, i, xy->km', mu, diff_L1[:, :, [2]], u, Wxy1, optimize=True)

    t7 = time.perf_counter()

    # Matriz jacobiana
    Jacob = K + diff_K_u_u - diff_K_u_alpha_x_x - K_u_alpha @ K_alpha_alpha_inv @ diff_K_alpha_u

    t8 = time.perf_counter()

    print(f"_pre_calculus: {t1 - t0:.6f} s")
    print(f"_pre_calculus2: {t2 - t1:.6f} s")
    print(f"condensação (K): {t3 - t2:.6f} s")
    print(f"diff_K_u_u: {t4 - t3:.6f} s")
    print(f"alpha: {t5 - t4:.6f} s")
    print(f"diff_K_u_alpha_x_x: {t6 - t5:.6f} s")
    print(f"diff_K_alpha_u: {t7 - t6:.6f} s")
    print(f"Jacob: {t8 - t7:.6f} s")
    print(f"tempo total K: {t8 - t0:.6f} s")

    return Jacob
