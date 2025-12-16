import numpy as np

from shellpy.fsdt6.transformation_matrix import transformation_matrix


def constitutive_matrix_in_shell_frame(mid_surface_geometry, C_local, position):

    T = transformation_matrix(mid_surface_geometry, position)
    if C_local.ndim == 2:
        C_shell_reciprocal_frame = np.einsum('ji...,jk,kl...->il...', T, C_local, T)
    else:
        C_shell_reciprocal_frame = np.einsum('ji...,jk...,kl...->il...', T, C_local, T)
    return C_shell_reciprocal_frame
