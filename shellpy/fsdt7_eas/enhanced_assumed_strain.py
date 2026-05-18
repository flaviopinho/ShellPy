import numpy as np

from ..displacement_expansion import DisplacementExpansion


def enhanced_assumed_strain(eas_field: DisplacementExpansion, integration_points, W_xy):
    xi1, xi2, xi3 = integration_points

    xx = 0

    shell_area = np.einsum('xy->', W_xy)
    n_dof = eas_field.number_of_degrees_of_freedom()
    M_zz = np.zeros((n_dof-xx, 1)+np.shape(xi1))
    for i in range(xx, n_dof):
        aux = eas_field.shape_function(i, xi1, xi2)
        if aux.ndim == 3:
            aux = aux[0]
        integral = np.einsum('xy, xy->', aux, W_xy)
        print(integral)
        M_zz[i-xx, 0] = aux #- integral*shell_area

    return M_zz

def enhanced_assumed_strain_full(eas_field, integration_points, W_xy, detW_z):
    xi1, xi2, xi3 = integration_points

    # Diferencial de volume completo
    dV = np.einsum('xy, xyz -> xyz', W_xy, detW_z)

    n_dof = eas_field.number_of_degrees_of_freedom()
    # M_zz volta a ser 2D (nx, ny)
    M_zz = np.zeros((n_dof, 1, xi1.shape[0], xi1.shape[1]))

    for i in range(0, n_dof):
        aux = eas_field.shape_function(i, xi1, xi2)
        if aux.ndim == 3:
            aux = aux[:, :, 0]

        # Numerador: integral( mu * xi3 * detJ )
        numerator = np.sum(aux[:, :, np.newaxis] * xi3 * dV)

        # Denominador: integral( xi3 * detJ )
        # Note que em placas isso tende a zero, em cascas é o efeito da curvatura
        denominator = np.sum(xi3 * dV)

        # Se o denominador for desprezível (placa pura), a correção é nula
        if abs(denominator) < 1e-12:
            correction = 0
        else:
            correction = numerator / denominator

        # Retorna a base espacial pura
        M_zz[i, 0] = aux - correction

    return M_zz