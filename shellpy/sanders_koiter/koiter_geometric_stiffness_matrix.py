from time import time
import numpy as np

from ..shell import Shell
from .koiter_strain_tensor import koiter_linear_strain_components, koiter_nonlinear_strain_components_total
from ..sanders_koiter.constitutive_tensor_koiter import plane_stress_constitutive_tensor_for_koiter_theory
from ..numeric_integration.boole_integral import boole_weights_simple_integral
from ..numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_z, \
    n_integral_default_y
from ..numeric_integration.integral_weights import double_integral_weights


def koiter_geometric_stiffness_matrix(shell: Shell, U0: np.ndarray,
                                           n_x=n_integral_default_x,
                                           n_y=n_integral_default_y, n_z=n_integral_default_z,
                                           integral_method=boole_weights_simple_integral):

    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)
    h = shell.thickness(xi1, xi2)
    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    n = np.shape(xi1)
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()


    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)
    C = plane_stress_constitutive_tensor_for_koiter_theory(shell.mid_surface_geometry, shell.material, xi1, xi2, xi3)


    C0 = np.einsum('ijklxyz, xyz, xyz, xyz->ijklxy', C, xi3 ** 0, det_shifter_tensor, Wz)
    C1 = np.einsum('ijklxyz, xyz, xyz, xyz->ijklxy', C, xi3 ** 1, det_shifter_tensor, Wz)

    Wxy = sqrtG * Wxy


    N0 = np.zeros((2, 2, n[0], n[1]))

    for i in range(n_dof):
            eps0_lin, eps1_lin = koiter_linear_strain_components(
                shell.mid_surface_geometry, shell.displacement_expansion, i, xi1, xi2)

            # L0 aqui representa o esforço resultante físico (N = C0*eps + C1*kappa)
            L0_lin_i = (np.einsum('abcdxy,cdxy->abxy', C0, eps0_lin) +
                        np.einsum('abcdxy,cdxy->abxy', C1, eps1_lin))
            N0 += U0[i] * L0_lin_i


    K_G = np.zeros((n_dof, n_dof))
    start = time()

    for i in range(n_dof):
        for j in range(i, n_dof):

            gamma_ij = koiter_nonlinear_strain_components_total(
                shell.mid_surface_geometry, shell.displacement_expansion, i, j, xi1, xi2)


            k_g_ij = 2.0 * np.einsum('abxy, abxy, xy -> ', N0, gamma_ij, Wxy, optimize=True)

            K_G[i, j] = k_g_ij
            if i != j:
                K_G[j, i] = k_g_ij

    print('Tempo de cálculo de K_G = ', time() - start)
    return K_G