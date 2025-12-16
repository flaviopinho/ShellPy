import numpy as np
from numpy.polynomial.legendre import Legendre

from shellpy import DisplacementExpansion


def enhanced_assumed_strain(eas_field: DisplacementExpansion, integration_points, W_xy, W_z):
    xi1, xi2, xi3 = integration_points

    shell_volume = np.einsum('xy, xyz->', W_xy, W_z)

    n_dof = eas_field.number_of_degrees_of_freedom()
    M_zz = np.zeros((n_dof-1, 1)+np.shape(xi3))
    for i in range(1, n_dof):
        aux = eas_field.shape_function(i, xi1, xi2)[0]
        aux = np.einsum('xy, xyz->xyz', aux, xi3)
        integral = np.einsum('xyz, xy, xyz->', aux, W_xy, W_z)
        M_zz[i-1, 0] = aux - integral*shell_volume

    return M_zz