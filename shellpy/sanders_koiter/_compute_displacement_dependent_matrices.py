import time
from functools import wraps
import numpy as np


def cache_last_u(func):
    """
    Decorator that caches the last evaluation based on the displacement vector `u`.

    This is essential for performance in non-linear solvers (like Newton-Raphson),
    preventing the recalculation of the Tangent Stiffness Matrix (K_T) if it is
    requested immediately after the Internal Force (F_int) at the exact same
    displacement step.
    """
    last_u = None
    last_result = None

    @wraps(func)
    def wrapper(u, *args, **kwargs):
        nonlocal last_u, last_result
        # If the displacement vector hasn't changed, return the cached result
        if last_u is not None and np.array_equal(u, last_u):
            return last_result

        # Otherwise, compute the function, update the cache, and return
        result = func(u, *args, **kwargs)
        last_u = u.copy()
        last_result = result
        return result

    return wrapper


@cache_last_u
def compute_displacement_dependent_matrices(u, Wxy1, C0, C1, C2, eps0_lin, eps1_lin, eps0_nl, eps1_nl):
    """
    Computes the tangent matrices that depend on the current displacement state.
    Returns both the Internal Force vector (Residual) and the complete
    Tangent Stiffness Matrix (Jacobian) simultaneously to maximize reuse of
    intermediate tensor contractions.

    Parameters
    ----------
    u : np.ndarray
        Current global displacement vector.
    Wxy1 : np.ndarray
        Differential area weights for numerical integration.
    C0, C1, C2 : np.ndarray
        Integrated constitutive tensors (Membrane, Coupling, Bending).
    eps0_lin, eps1_lin : np.ndarray
        Linear strain tensors (membrane and curvature).
    eps0_nl, eps1_nl : np.ndarray
        Non-linear (quadratic) strain tensors.

    Returns
    -------
    F_int : np.ndarray
        Internal force vector.
    Jacobian : np.ndarray
        Complete Tangent Stiffness Matrix (Material + Geometric).
    """
    t0 = time.perf_counter()

    # --------------------------------------------------------------------------------
    # 1. CURRENT STRAINS AND STRESS RESULTANTS
    # --------------------------------------------------------------------------------
    # Evaluate current membrane strains (e0) and curvature changes (e1)
    # Total Strain = Linear Part (B_lin * u) + Non-linear Part (u^T * H * u)
    e0_curr = np.einsum('m, maxy->axy', u, eps0_lin, optimize=True) + \
              np.einsum('m, n, mnaxy->axy', u, u, eps0_nl, optimize=True)

    e1_curr = np.einsum('m, maxy->axy', u, eps1_lin, optimize=True) + \
              np.einsum('m, n, mnaxy->axy', u, u, eps1_nl, optimize=True)

    # Evaluate current stress resultants using the constitutive relations
    # Membrane forces (N) and Bending moments (M)
    N = np.einsum('ijxy, jxy->ixy', C0, e0_curr, optimize=True) + \
        np.einsum('ijxy, jxy->ixy', C1, e1_curr, optimize=True)

    M = np.einsum('ijxy, jxy->ixy', C1, e0_curr, optimize=True) + \
        np.einsum('ijxy, jxy->ixy', C2, e1_curr, optimize=True)

    # --------------------------------------------------------------------------------
    # 2. TANGENT KINEMATIC MATRICES (B-Matrices)
    # --------------------------------------------------------------------------------
    # The B-matrix relates strain increments to displacement increments: d(eps) = B * du
    # B = B_linear + 2 * B_nonlinear(u)
    B0 = eps0_lin + 2.0 * np.einsum('mnaxy, n->maxy', eps0_nl, u, optimize=True)
    B1 = eps1_lin + 2.0 * np.einsum('mnaxy, n->maxy', eps1_nl, u, optimize=True)

    # --------------------------------------------------------------------------------
    # 3. INTERNAL FORCE VECTOR (RESIDUAL)
    # --------------------------------------------------------------------------------
    # F_int = Integral( B^T * sigma ) dA
    # This represents the internal restoring forces of the shell.
    F_int = np.einsum('maxy, axy, xy->m', B0, N, Wxy1, optimize=True) + \
            np.einsum('maxy, axy, xy->m', B1, M, Wxy1, optimize=True)

    # --------------------------------------------------------------------------------
    # 4. TANGENT STIFFNESS MATRIX (JACOBIAN)
    # --------------------------------------------------------------------------------

    # --- 4.1 Material Stiffness Matrix (K_M) ---
    # Accounts for changes in stress due to changes in strain: Integral( B^T * C * B ) dA

    # First, compute the change in stress resultants with respect to displacements
    N_dn = np.einsum('ijxy, njxy->nixy', C0, B0, optimize=True) + \
           np.einsum('ijxy, njxy->nixy', C1, B1, optimize=True)

    M_dn = np.einsum('ijxy, njxy->nixy', C1, B0, optimize=True) + \
           np.einsum('ijxy, njxy->nixy', C2, B1, optimize=True)

    # Then, assemble the Material Stiffness matrix
    K_M = np.einsum('maxy, naxy, xy->mn', B0, N_dn, Wxy1, optimize=True) + \
          np.einsum('maxy, naxy, xy->mn', B1, M_dn, Wxy1, optimize=True)

    # --- 4.2 Geometric Stiffness Matrix (K_G) ---
    # Also known as the Initial Stress matrix. It accounts for the change in the
    # B-matrix due to displacements, scaled by the current stress state.
    K_G = 2.0 * (np.einsum('mnaxy, axy, xy->mn', eps0_nl, N, Wxy1, optimize=True) + \
                 np.einsum('mnaxy, axy, xy->mn', eps1_nl, M, Wxy1, optimize=True))

    # Total Tangent Stiffness (Jacobian)
    Jacobian = K_M + K_G

    # Optional diagnostic print for performance profiling
    # print(f"Update state time: {time.perf_counter() - t0:.5f}s")

    return F_int, Jacobian