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


def enhanced_assumed_strain_full(eas_field: DisplacementExpansion, integration_points, W_xy, detW_z):
    xi1, xi2, xi3 = integration_points

    shell_volume = np.einsum('xy, xyz->', W_xy, detW_z)
    n_dof = eas_field.number_of_degrees_of_freedom()
    M_zz = np.zeros((n_dof, 1)+np.shape(xi1))
    for i in range(n_dof):
        aux = eas_field.shape_function(i, xi1, xi2)
        if aux.ndim == 3:
            aux = aux[0]

        integral = np.einsum('xy, xyz->', aux*W_xy,xi3*detW_z)
        #print(integral)
        M_zz[i, 0] = aux - integral*shell_volume

    return M_zz