import numpy as np

from shellpy.sanders_koiter.plane_stress_transformation_matrix import plane_stress_transformation_matrix


def plane_stress_constitutive_matrix_in_shell_frame(mid_surface_geometry, C_local, position):
    """
    Transforms the 3x3 plane-stress constitutive matrix from the local material
    frame to the shell's reciprocal curvilinear frame.

    This operation performs a congruent transformation (T^T * C * T) to ensure
    that the stiffness properties are correctly aligned with the coordinates
    used for strain calculations.

    Parameters:
    -----------
    mid_surface_geometry : MidSurfaceGeometry
        The geometric definition of the shell surface.
    C_local : np.ndarray
        The 3x3 constitutive matrix in Voigt notation [11, 22, 12] defined
        in the material frame.
    position : tuple
        The (xi1, xi2, xi3) coordinates where the transformation is evaluated.

    Returns:
    --------
    C_shell_reciprocal_frame : np.ndarray
        The transformed constitutive matrix in the shell coordinate system.
    """

    # --- 1. Compute Transformation Matrix ---
    # Obtain the Voigt-compatible transformation matrix T at the given position.
    # T relates material strains to shell strains: eps_material = T * eps_shell
    T = plane_stress_transformation_matrix(mid_surface_geometry, position)

    # --- 2. Perform Congruent Transformation ---
    # The transformation follows the rule: C_shell = T.T @ C_local @ T
    # We use np.einsum to handle vectorized integration points (multi-dimensional arrays).

    if C_local.ndim == 2:
        # Case A: C_local is constant across the domain (e.g., homogeneous material).
        # 'ji...' represents T_transpose, 'jk' is C_local, 'kl...' is T.
        C_shell_reciprocal_frame = np.einsum('ji...,jk,kl...->il...', T, C_local, T)
    else:
        # Case B: C_local varies spatially (e.g., FGM or spatially varying properties).
        # Here, C_local also has spatial dimensions (...) matching T.
        C_shell_reciprocal_frame = np.einsum('ji...,jk...,kl...->il...', T, C_local, T)

    return C_shell_reciprocal_frame