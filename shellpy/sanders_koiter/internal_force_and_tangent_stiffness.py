import time
from shellpy import Shell
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
from shellpy.numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_y, \
    n_integral_default_z

from ._compute_constant_shell_matrices import compute_constant_shell_matrices
from ._compute_displacement_dependent_matrices import compute_displacement_dependent_matrices


def internal_force_vector(u, shell: Shell,
                          n_x=n_integral_default_x,
                          n_y=n_integral_default_y,
                          n_z=n_integral_default_z,
                          integral_method=gauss_weights_simple_integral):
    """
    Computes the internal force vector (residual) for the Sanders-Koiter shell formulation.

    This function acts as the primary interface for non-linear solvers. It utilizes
    a smart caching mechanism under the hood to ensure high performance during
    Newton-Raphson or Arc-Length (Continuation) routines.

    Parameters
    ----------
    u : np.ndarray
        The current global displacement vector (modal amplitudes).
    shell : Shell
        The shell object containing geometry, material, and expansion definitions.
    n_x, n_y, n_z : int, optional
        Number of integration points.
    integral_method : function, optional
        Numerical integration scheme.

    Returns
    -------
    F_int : np.ndarray
        The internal force vector evaluated at displacement 'u'.
    """
    t0 = time.perf_counter()

    # 1. Retrieve the static (cached) matrices
    # These matrices depend only on the shell geometry and material, not on displacements.
    # The @cache_function decorator ensures this expensive step runs only once per mesh.
    Wxy1, C0, C1, C2, eps0_lin, eps1_lin, eps0_nl, eps1_nl = compute_constant_shell_matrices(
        shell, n_x, n_y, n_z, integral_method
    )

    # 2. Compute the displacement-dependent state
    # This evaluates the non-linear kinematics and constitutive relations.
    F_int, _ = compute_displacement_dependent_matrices(
        u, Wxy1, C0, C1, C2, eps0_lin, eps1_lin, eps0_nl, eps1_nl
    )

    # print(f"Internal Force Time: {time.perf_counter() - t0:.6f} s")

    return F_int


def tangent_stiffness_matrix(u, shell: Shell,
                             n_x=n_integral_default_x,
                             n_y=n_integral_default_y,
                             n_z=n_integral_default_z,
                             integral_method=gauss_weights_simple_integral):
    """
    Computes the Tangent Stiffness Matrix (Jacobian) for the Sanders-Koiter formulation.

    In a standard non-linear solver, the stiffness matrix is often evaluated immediately
    after the internal force vector. Because the underlying function uses a `@cache_last_u`
    decorator, if `u` has not changed since the `internal_force_vector` call, this
    function will return the Jacobian nearly instantaneously.

    Parameters
    ----------
    u : np.ndarray
        The current global displacement vector.
    shell : Shell
        The shell object.

    Returns
    -------
    Jacobian : np.ndarray
        The tangent stiffness matrix evaluated at displacement 'u'.
    """
    t0 = time.perf_counter()

    # 1. Retrieve the static (cached) matrices
    Wxy1, C0, C1, C2, eps0_lin, eps1_lin, eps0_nl, eps1_nl = compute_constant_shell_matrices(
        shell, n_x, n_y, n_z, integral_method
    )

    # 2. Compute the displacement-dependent state (or fetch from cache)
    # If this is called right after `internal_force_vector` with the same 'u',
    # the underlying function bypasses the math and immediately returns the cached Jacobian.
    _, Jacobian = compute_displacement_dependent_matrices(
        u, Wxy1, C0, C1, C2, eps0_lin, eps1_lin, eps0_nl, eps1_nl
    )

    # print(f"Tangent Stiffness Time: {time.perf_counter() - t0:.6f} s")

    return Jacobian