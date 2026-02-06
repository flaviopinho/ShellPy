import numpy as np
from numpy.polynomial.legendre import Legendre

from shellpy import DisplacementExpansion


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


def enhanced_assumed_strain_old(eas_field: DisplacementExpansion, integration_points, W_xy, W_z):
    xi1, xi2, xi3 = integration_points

    shell_volume = np.einsum('xy, xyz->', W_xy, W_z)
    n_dof = eas_field.number_of_degrees_of_freedom()
    M_zz = np.zeros((n_dof, 1)+np.shape(xi1))
    for i in range(n_dof):
        aux = eas_field.shape_function(i, xi1, xi2)[0]
        aux2 = np.einsum('xy, xyz->xyz', aux, xi3)
        integral = np.einsum('xyz, xy, xyz->', aux, W_xy, W_z)
        print(integral)
        M_zz[i, 0] = aux - integral*shell_volume

    return M_zz