import numpy as np
from shellpy import MidSurfaceGeometry


def transformation_matrix_fosd2_global(mid_surface_geometry: MidSurfaceGeometry, xi1, xi2, xi3):

    xi1 = np.atleast_1d(xi1)
    xi2 = np.atleast_1d(xi2)
    xi3 = np.atleast_1d(xi3)

    # Mid-surface basis
    natural_base = mid_surface_geometry.natural_base(xi1, xi2)
    reciprocal_base = mid_surface_geometry.reciprocal_base(xi1, xi2)

    # Material directions in global frame
    e1_material = np.array((1, 0, 0))
    e2_material = np.array((0, 1, 0))
    e3_material = np.array((0, 0, 1))
    material_base = np.stack((e1_material, e2_material, e3_material), axis=0)

    # Obtain the inverse shifter tensor (3x3 in-plane approximation)
    inverse_shift_tensor = mid_surface_geometry.shifter_tensor_inverse_approximation(xi1, xi2, xi3)
    inverse_shift_tensor_extended = np.zeros((3, 3) + xi1.shape + xi3.shape)
    inverse_shift_tensor_extended[0:2, 0:2] = inverse_shift_tensor
    inverse_shift_tensor_extended[2, 2] = 1

    # Convert tuples/lists to 3x3 arrays
    reciprocal_base = np.stack(reciprocal_base, axis=0)
    reciprocal_base = reciprocal_base.reshape((3, 3) + xi1.shape)

    # Determine transformation_matrix
    transformation_matrix = np.einsum('ik, lj...z,lk...->ij...z',
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

    auxiliar_tensor = auxiliar_tensor.reshape(9, 9, -1)

    auxiliar_tensor = auxiliar_tensor.reshape((9, 9) + xi1.shape + xi3.shape)

    transformation_matrix = np.einsum('ij,jk...,kl->il...',
                                      permutation_voigt,
                                      auxiliar_tensor,
                                      inverse_permutation_voigt)

    return transformation_matrix


def transformation_matrix_fosd2_alpha(mid_surface_geometry: MidSurfaceGeometry, alpha, xi1, xi2, xi3):

    alpha = np.atleast_1d(alpha)
    xi1 = np.atleast_1d(xi1)
    xi2 = np.atleast_1d(xi2)
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

    # Angle mesured from natural_base[0]
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    e1_material_rot = np.einsum('i...,z->i...z', e1_material, cos_a) + np.einsum('i...,z->i...z', e2_material, sin_a)
    e2_material_rot = np.einsum('i...,z->i...z', e1_material, -sin_a) + np.einsum('i...,z->i...z', e2_material, cos_a)
    e3_material_rot = np.repeat(e3_material[..., None], xi3.size, axis=-1)

    e1_material_rot = e1_material_rot.reshape((3,) + xi1.shape + xi3.shape)
    e2_material_rot = e2_material_rot.reshape((3,) + xi1.shape + xi3.shape)
    e3_material_rot = e3_material_rot.reshape((3,) + xi1.shape + xi3.shape)

    material_base = np.stack((e1_material_rot, e2_material_rot, e3_material_rot), axis=0)

    # Obtain the inverse shifter tensor (3x3 in-plane approximation)
    inverse_shift_tensor = mid_surface_geometry.shifter_tensor_inverse_approximation(xi1, xi2, xi3)
    inverse_shift_tensor_extended = np.zeros((3, 3) + xi1.shape + xi3.shape)
    inverse_shift_tensor_extended[0:2, 0:2] = inverse_shift_tensor
    inverse_shift_tensor_extended[2, 2] = 1

    # Convert tuples/lists to 3x3 arrays
    reciprocal_base = np.stack(reciprocal_base, axis=0)
    reciprocal_base = reciprocal_base.reshape((3, 3) + xi1.shape)

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

    auxiliar_tensor = auxiliar_tensor.reshape(9, 9, -1)

    auxiliar_tensor = auxiliar_tensor.reshape((9, 9)+xi1.shape+xi3.shape)

    transformation_matrix = np.einsum('ij,jk...,kl->il...',
                                      permutation_voigt,
                                      auxiliar_tensor,
                                      inverse_permutation_voigt)

    return transformation_matrix
