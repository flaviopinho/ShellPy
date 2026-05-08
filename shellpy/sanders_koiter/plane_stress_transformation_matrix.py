import numpy as np

from ..midsurface_geometry import MidSurfaceGeometry

# -------------------------------------------------------------------------
# 2D Permutation Matrices for Classic Voigt Order [11, 22, 12]
# Flattened tensor vector: [11, 12, 21, 22] (indices 0, 1, 2, 3)
# Voigt Vector: [e11, e22, gamma12] (where gamma12 = e12 + e21)
# These matrices automate the conversion between 2nd-order tensors and Voigt vectors.
# -------------------------------------------------------------------------
PERMUTATION_VOIGT_2D = np.zeros((3, 4))
PERMUTATION_VOIGT_2D[0, 0] = 1.0  # Maps tensor 11 -> Voigt e11
PERMUTATION_VOIGT_2D[1, 3] = 1.0  # Maps tensor 22 -> Voigt e22
PERMUTATION_VOIGT_2D[2, 1] = 1.0  # Maps tensor 12 -> Voigt gamma12
PERMUTATION_VOIGT_2D[2, 2] = 1.0  # Maps tensor 21 -> Voigt gamma12

INVERSE_PERMUTATION_VOIGT_2D = np.zeros((4, 3))
INVERSE_PERMUTATION_VOIGT_2D[0, 0] = 1.0  # Maps Voigt e11 -> tensor 11
INVERSE_PERMUTATION_VOIGT_2D[1, 2] = 0.5  # Maps Voigt gamma12 / 2 -> tensor 12
INVERSE_PERMUTATION_VOIGT_2D[2, 2] = 0.5  # Maps Voigt gamma12 / 2 -> tensor 21
INVERSE_PERMUTATION_VOIGT_2D[3, 1] = 1.0  # Maps Voigt e22 -> tensor 22


def plane_stress_transformation_matrix(mid_surface_geometry: MidSurfaceGeometry, position):
    """
    Computes the 3x3 transformation matrix in Voigt notation to map strains
    from the shell frame (curvilinear) to the material frame (local orthogonal).

    Parameters:
    -----------
    mid_surface_geometry : MidSurfaceGeometry
        Geometric definition of the shell mid-surface.
    position : tuple
        Coordinates (xi1, xi2, xi3) where the matrix is evaluated.

    Returns:
    --------
    transformation_matrix_voigt : np.ndarray
        A 3x3 matrix relating Voigt strain vectors: eps_material = T * eps_shell.
    """
    xi1 = position[0]
    xi2 = position[1]
    xi3 = position[2]

    # Handle input shapes for vectorized integration points
    shape_xi3 = np.shape(xi3)
    last_dim = (shape_xi3[-1],) if shape_xi3 != () else ()
    shape_init = np.shape(xi1) + last_dim

    xi1 = np.atleast_2d(xi1)
    xi2 = np.atleast_2d(xi2)
    xi3 = np.atleast_1d(xi3)

    # Acquire geometric bases
    natural_base = mid_surface_geometry.natural_base(xi1, xi2)
    reciprocal_base = mid_surface_geometry.reciprocal_base(xi1, xi2)

    # Material frame vectors (usually orthogonal and normalized)
    e1_material = natural_base[0]
    e2_material = reciprocal_base[1]
    e3_material = natural_base[2]

    def normalize(v, axis=0):
        norm = np.linalg.norm(v, axis=axis, keepdims=True)
        return v / norm

    e1_material = normalize(e1_material, axis=0)
    e2_material = normalize(e2_material, axis=0)
    e3_material = normalize(e3_material, axis=0)

    # Material base orientation tensor
    material_base = np.stack((e1_material, e2_material, e3_material), axis=0)

    # Account for shell thickness shifting (Shifter Tensor / Parallel Surface mapping)
    inverse_shift_tensor_extended = mid_surface_geometry.shifter_tensor_inverse_extended(xi1, xi2, xi3)
    inverse_shift_tensor_extended = inverse_shift_tensor_extended.reshape((3, 3) + xi1.shape + (xi3.shape[-1],))

    reciprocal_base = np.stack(reciprocal_base, axis=0)

    # --- 3D Tensor Transformation ---
    # Computes the mapping between frames for a full 3rd-order tensor
    transformation_matrix_3d = np.einsum('ik...,lj...z, lk...->ij...z',
                                         material_base,
                                         inverse_shift_tensor_extended,
                                         reciprocal_base)

    # Extract the 2x2 in-plane components for the Plane Stress assumption
    transformation_matrix_2d = transformation_matrix_3d[0:2, 0:2, ...]

    # Flatten the 2x2 transformation to a 4x4 representation to handle Voigt indices
    auxiliar_tensor = np.einsum("ab...,cd...->acbd...",
                                transformation_matrix_2d,
                                transformation_matrix_2d)
    shape_rest = auxiliar_tensor.shape[4:]
    auxiliar_tensor = auxiliar_tensor.reshape(4, 4, *shape_rest)

    # Convert the 4x4 tensor mapping into the 3x3 Voigt mapping
    transformation_matrix_voigt = np.einsum('ij,jk...,kl->il...',
                                            PERMUTATION_VOIGT_2D,
                                            auxiliar_tensor,
                                            INVERSE_PERMUTATION_VOIGT_2D)

    return np.reshape(transformation_matrix_voigt, (3, 3) + shape_init)


def plane_stress_transformation_matrix_local(mid_surface_geometry: MidSurfaceGeometry, xi1, xi2, xi3, alpha):
    """
    Similar to plane_stress_transformation_matrix, but includes a local rotation
    angle 'alpha' (e.g., for fiber orientation in laminate composites).

    Parameters:
    -----------
    alpha : float or np.ndarray
        Fiber orientation angle in radians.
    """
    xi1 = np.atleast_2d(xi1)
    xi2 = np.atleast_2d(xi2)
    xi3 = np.atleast_1d(xi3)
    alpha = np.atleast_1d(alpha)

    natural_base = mid_surface_geometry.natural_base(xi1, xi2)
    reciprocal_base = mid_surface_geometry.reciprocal_base(xi1, xi2)

    e1_material = natural_base[0]
    e2_material = reciprocal_base[1]
    e3_material = natural_base[2]

    def normalize(v, axis=0):
        norm = np.linalg.norm(v, axis=axis, keepdims=True)
        return v / norm

    e1_material = normalize(e1_material, axis=0)
    e2_material = normalize(e2_material, axis=0)

    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    # Rotate the local material base by the angle alpha
    e1_material_rot = np.einsum('i...,...z->i...z', e1_material, cos_a) + np.einsum('i...,...z->i...z', e2_material,
                                                                                    sin_a)
    e2_material_rot = np.einsum('i...,...z->i...z', e1_material, -sin_a) + np.einsum('i...,...z->i...z', e2_material,
                                                                                     cos_a)
    e3_material_rot = np.repeat(e3_material[..., None], xi3.shape[-1], axis=-1)

    material_base = np.stack((e1_material_rot, e2_material_rot, e3_material_rot), axis=0)

    inverse_shift_tensor_extended = mid_surface_geometry.shifter_tensor_inverse_extended(xi1, xi2, xi3)
    inverse_shift_tensor_extended = inverse_shift_tensor_extended.reshape((3, 3) + xi1.shape + (xi3.shape[-1],))

    reciprocal_base = np.stack(reciprocal_base, axis=0)

    # Complete 3D transformation with rotation
    transformation_matrix_3d = np.einsum('ik...z,lj...z, lk...->ij...z',
                                         material_base,
                                         inverse_shift_tensor_extended,
                                         reciprocal_base)

    # Extract 2x2 in-plane component
    transformation_matrix_2d = transformation_matrix_3d[0:2, 0:2, ...]

    # Convert to Voigt 3x3
    auxiliar_tensor = np.einsum("ab...,cd...->acbd...",
                                transformation_matrix_2d,
                                transformation_matrix_2d)
    shape_rest = auxiliar_tensor.shape[4:]
    auxiliar_tensor = auxiliar_tensor.reshape(4, 4, *shape_rest)

    transformation_matrix_voigt = np.einsum('ij,jk...,kl->il...',
                                            PERMUTATION_VOIGT_2D,
                                            auxiliar_tensor,
                                            INVERSE_PERMUTATION_VOIGT_2D)

    return np.squeeze(transformation_matrix_voigt)


def plane_stress_transformation_matrix_rotation(alpha):
    """
    Exact analytical solution for the strain transformation matrix (Voigt 3x3)
    under a simple coordinate system rotation by an angle 'alpha'.

    Relates rotated strains to original strains: eps' = T * eps.
    """
    c = np.cos(alpha)
    s = np.sin(alpha)
    c2, s2, cs = c ** 2, s ** 2, c * s

    # 3x3 matrix for Voigt strain [e11, e22, gamma12]
    T = np.array([
        [c2, s2, cs],
        [s2, c2, -cs],
        [-2 * cs, 2 * cs, c2 - s2]
    ])

    return T