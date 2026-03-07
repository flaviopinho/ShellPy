import numpy as np
import time
from shellpy import Shell
from shellpy.fsdt7_eas._pre_calculus import _pre_calculus, _pre_calculus2
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral


def internal_force_vector(u, shell: Shell, eas_field, n_x=20, n_y=20, n_z=10,
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

    F_int = K @ u

    t4 = time.perf_counter()

    print(f"_pre_calculus: {t1 - t0:.6f} s")
    print(f"_pre_calculus2: {t2 - t1:.6f} s")
    print(f"condensação (K): {t3 - t2:.6f} s")
    print(f"F_int: {t4 - t3:.6f} s")
    print(f"tempo total F: {t4 - t0:.6f} s")

    return F_int
