import numpy as np
from shellpy import MidSurfaceGeometry


def transformation_matrix(mid_surface_geometry: MidSurfaceGeometry, position):

    xi1 = position[0]
    xi2 = position[1]
    xi3 = position[2]

    shape_xi3 = np.shape(xi3)
    last_dim = (shape_xi3[-1],) if shape_xi3 != () else ()
    shape_init = np.shape(xi1) + last_dim

    xi1 = np.atleast_2d(xi1)
    xi2 = np.atleast_2d(xi2)
    xi3 = np.atleast_1d(xi3)

    # Mid-surface basis
    natural_base = mid_surface_geometry.natural_base(xi1, xi2)  # shape (3, ...xi1.shape)
    reciprocal_base = mid_surface_geometry.reciprocal_base(xi1, xi2)

    # Material basis
    e1_material = natural_base[0]  # shape (3, ...xi1.shape)
    e2_material = reciprocal_base[1]
    e3_material = natural_base[2]

    def normalize(v, axis=0):
        norm = np.linalg.norm(v, axis=axis, keepdims=True)
        return v / norm

    e1_material = normalize(e1_material, axis=0)
    e2_material = normalize(e2_material, axis=0)
    e3_material = normalize(e3_material, axis=0)

    material_base = np.stack((e1_material, e2_material, e3_material), axis=0)

    # Obtain the inverse shifter tensor (3x3 in-plane approximation)
    inverse_shift_tensor_extended = mid_surface_geometry.shifter_tensor_inverse_extended(xi1, xi2, xi3)
    inverse_shift_tensor_extended = inverse_shift_tensor_extended.reshape((3, 3) + xi1.shape + (xi3.shape[-1],))

    # Convert tuples/lists to 3x3 arrays
    reciprocal_base = np.stack(reciprocal_base, axis=0)

    # Determine transformation_matrix
    transformation_matrix = np.einsum('ik...,lj...z, lk...->ij...z',
                                      material_base,
                                      inverse_shift_tensor_extended,
                                      reciprocal_base
                                      )

    # permutation matrix: transformation from 9x1 vector to 6x1 Voigt notation
    permutation_voigt = np.zeros((6, 9))
    permutation_voigt[0, 0] = 1
    permutation_voigt[1, 4] = 1
    permutation_voigt[2, 8] = 1
    permutation_voigt[3, 5] = 1
    permutation_voigt[3, 7] = 1
    permutation_voigt[4, 2] = 1
    permutation_voigt[4, 6] = 1
    permutation_voigt[5, 1] = 1
    permutation_voigt[5, 3] = 1

    # inverse permutation matrix: transformation from 6x1 Voigt to 9x1 vector
    inverse_permutation_voigt = np.zeros((9, 6))
    inverse_permutation_voigt[0, 0] = 1
    inverse_permutation_voigt[1, 5] = 0.5
    inverse_permutation_voigt[2, 4] = 0.5
    inverse_permutation_voigt[3, 5] = 0.5
    inverse_permutation_voigt[4, 1] = 1
    inverse_permutation_voigt[5, 3] = 0.5
    inverse_permutation_voigt[6, 4] = 0.5
    inverse_permutation_voigt[7, 3] = 0.5
    inverse_permutation_voigt[8, 2] = 1

    auxiliar_tensor = np.einsum("ab...,cd...->acbd...",
                                transformation_matrix,
                                transformation_matrix)
    shape_rest = auxiliar_tensor.shape[4:]
    auxiliar_tensor = auxiliar_tensor.reshape(9, 9, *shape_rest)

    transformation_matrix = np.einsum('ij,jk...,kl->il...',
                                      permutation_voigt,
                                      auxiliar_tensor,
                                      inverse_permutation_voigt)

    return np.reshape(transformation_matrix, (6,6)+shape_init)


def transformation_matrix_local(mid_surface_geometry: MidSurfaceGeometry, xi1, xi2, xi3, alpha):

    xi1 = np.atleast_2d(xi1)
    xi2 = np.atleast_2d(xi2)
    xi3 = np.atleast_1d(xi3)
    alpha = np.atleast_1d(alpha)

    # Mid-surface basis
    natural_base = mid_surface_geometry.natural_base(xi1, xi2)  # shape (3, ...xi1.shape)
    reciprocal_base = mid_surface_geometry.reciprocal_base(xi1, xi2)

    # Material basis
    e1_material = natural_base[0]  # shape (3, ...xi1.shape)
    e2_material = reciprocal_base[1]
    e3_material = natural_base[2]

    def normalize(v, axis=0):
        norm = np.linalg.norm(v, axis=axis, keepdims=True)
        return v / norm

    e1_material = normalize(e1_material, axis=0)
    e2_material = normalize(e2_material, axis=0)
    e3_material = normalize(e3_material, axis=0)

    # Angle mesured from natural_base[0]
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    e1_material_rot = np.einsum('i...,...z->i...z', e1_material, cos_a) + np.einsum('i...,...z->i...z', e2_material, sin_a)
    e2_material_rot = np.einsum('i...,...z->i...z', e1_material, -sin_a) + np.einsum('i...,...z->i...z', e2_material, cos_a)
    e3_material_rot = np.repeat(e3_material[..., None], xi3.shape[-1], axis=-1)

    material_base = np.stack((e1_material_rot, e2_material_rot, e3_material_rot), axis=0)

    # Obtain the inverse shifter tensor (3x3 in-plane approximation)
    inverse_shift_tensor_extended = mid_surface_geometry.shifter_tensor_inverse_extended(xi1, xi2, xi3)
    inverse_shift_tensor_extended = inverse_shift_tensor_extended.reshape((3, 3)+xi1.shape+(xi3.shape[-1],))

    # Convert tuples/lists to 3x3 arrays
    reciprocal_base = np.stack(reciprocal_base, axis=0)

    # Determine transformation_matrix
    transformation_matrix = np.einsum('ik...z,lj...z, lk...->ij...z',
                                      material_base,
                                      inverse_shift_tensor_extended,
                                      reciprocal_base
                                      )

    # permutation matrix: transformation from 9x1 vector to 6x1 Voigt notation
    permutation_voigt = np.zeros((6, 9))
    permutation_voigt[0, 0] = 1
    permutation_voigt[1, 4] = 1
    permutation_voigt[2, 8] = 1
    permutation_voigt[3, 5] = 1
    permutation_voigt[3, 7] = 1
    permutation_voigt[4, 2] = 1
    permutation_voigt[4, 6] = 1
    permutation_voigt[5, 1] = 1
    permutation_voigt[5, 3] = 1

    # inverse permutation matrix: transformation from 6x1 Voigt to 9x1 vector
    inverse_permutation_voigt = np.zeros((9, 6))
    inverse_permutation_voigt[0, 0] = 1
    inverse_permutation_voigt[1, 5] = 0.5
    inverse_permutation_voigt[2, 4] = 0.5
    inverse_permutation_voigt[3, 5] = 0.5
    inverse_permutation_voigt[4, 1] = 1
    inverse_permutation_voigt[5, 3] = 0.5
    inverse_permutation_voigt[6, 4] = 0.5
    inverse_permutation_voigt[7, 3] = 0.5
    inverse_permutation_voigt[8, 2] = 1

    auxiliar_tensor = np.einsum("ab...,cd...->acbd...",
                                transformation_matrix,
                                transformation_matrix)
    shape_rest = auxiliar_tensor.shape[4:]
    auxiliar_tensor = auxiliar_tensor.reshape(9, 9, *shape_rest)

    transformation_matrix = np.einsum('ij,jk...,kl->il...',
                                      permutation_voigt,
                                      auxiliar_tensor,
                                      inverse_permutation_voigt)

    return np.squeeze(transformation_matrix)



def transformation_matrix_rotation(alpha):
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])

    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    e1_material = e1 * cos_a + e2 * sin_a
    e2_material = -e1 * sin_a + e2 * cos_a
    e3_material = e3

    material_base = np.stack((e1_material, e2_material, e3_material), axis=0)

    # Convert tuples/lists to 3x3 arrays
    shell_local_base = np.stack((e1, e2, e3), axis=0)

    # Determine transformation_matrix
    transformation_matrix = np.einsum('ik,jk->ij',
                                      material_base, shell_local_base)

    # permutation matrix: transformation from 9x1 vector to 6x1 Voigt notation
    permutation_voigt = np.zeros((6, 9))
    permutation_voigt[0, 0] = 1
    permutation_voigt[1, 4] = 1
    permutation_voigt[2, 8] = 1
    permutation_voigt[3, 5] = 1
    permutation_voigt[3, 7] = 1
    permutation_voigt[4, 2] = 1
    permutation_voigt[4, 6] = 1
    permutation_voigt[5, 1] = 1
    permutation_voigt[5, 3] = 1

    # inverse permutation matrix: transformation from 6x1 Voigt to 9x1 vector
    inverse_permutation_voigt = np.zeros((9, 6))
    inverse_permutation_voigt[0, 0] = 1
    inverse_permutation_voigt[1, 5] = 0.5
    inverse_permutation_voigt[2, 4] = 0.5
    inverse_permutation_voigt[3, 5] = 0.5
    inverse_permutation_voigt[4, 1] = 1
    inverse_permutation_voigt[5, 3] = 0.5
    inverse_permutation_voigt[6, 4] = 0.5
    inverse_permutation_voigt[7, 3] = 0.5
    inverse_permutation_voigt[8, 2] = 1

    auxiliar_tensor = np.einsum("ab,cd->acbd",
                                transformation_matrix,
                                transformation_matrix)
    shape_rest = auxiliar_tensor.shape[4:]
    auxiliar_tensor = auxiliar_tensor.reshape(9, 9, *shape_rest)

    transformation_matrix = np.einsum('ij,jk,kl->il',
                                      permutation_voigt,
                                      auxiliar_tensor,
                                      inverse_permutation_voigt)

    return transformation_matrix
