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
    """
    Computes the Geometric Stiffness Matrix (K_G) for a shell structure
    using the Koiter approximation, evaluated at a fundamental pre-buckling state U0.

    Parameters:
    - shell (Shell): The shell object containing material, geometry, etc.
    - U0 (ndarray): The pre-buckling displacement vector (solution of K_E * U0 = F).
    - n_x, n_y, n_z (int): Number of integration points along each coordinate direction.
    - integral_method (function): Integration method used for numerical integration.

    Returns:
    - K_G (ndarray): The geometric stiffness matrix (n_dof x n_dof).
    """

    # 1. Configuração de pontos e pesos de integração
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)
    h = shell.thickness(xi1, xi2)
    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    n = np.shape(xi1)
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # 2. Grandezas geométricas e tensor constitutivo
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)
    C = plane_stress_constitutive_tensor_for_koiter_theory(shell.mid_surface_geometry, shell.material, xi1, xi2, xi3)

    C0 = 1 / 2 * np.einsum('ijklxyz, xyz, xyz, xyz->ijklxy', C, xi3 ** 0, det_shifter_tensor, Wz)
    C1 = 1 / 2 * np.einsum('ijklxyz, xyz, xyz, xyz->ijklxy', C, xi3 ** 1, det_shifter_tensor, Wz)

    Wxy = sqrtG * Wxy

    # 3. Cálculo do Estado Fundamental (Esforços N0)
    N0 = np.zeros((2, 2, n[0], n[1]))

    print('Calculando os esforços de membrana do estado fundamental (N0)...')
    for i in range(n_dof):
        # Otimização: ignorar DOFs que não foram ativados no estado linear
        if abs(U0[i]) > 1e-12:
            eps0_lin, eps1_lin = koiter_linear_strain_components(
                shell.mid_surface_geometry, shell.displacement_expansion, i, xi1, xi2)

            # L0 representa os esforços resultantes de membrana para o DOF i
            L0_lin_i = (np.einsum('abcdxy,cdxy->abxy', C0, eps0_lin) +
                        np.einsum('abcdxy,cdxy->abxy', C1, eps1_lin))

            # Acumulando a contribuição deste grau de liberdade aos esforços globais pré-flambagem
            N0 += U0[i] * L0_lin_i

    # 4. Montagem da Matriz de Rigidez Geométrica
    K_G = np.zeros((n_dof, n_dof))

    print('Montando a Matriz de Rigidez Geométrica (K_G)...')
    start = time()

    for i in range(n_dof):
        for j in range(i, n_dof):
            # Deformação não-linear cruzada para os DOFs i e j
            gamma_ij = koiter_nonlinear_strain_components_total(
                shell.mid_surface_geometry, shell.displacement_expansion, i, j, xi1, xi2)

            # A integral do Trabalho Virtual: integral( N0 : gamma_ij * dA )
            k_g_ij = np.einsum('abxy, abxy, xy -> ', N0, gamma_ij, Wxy, optimize=True)

            # Preenchimento espelhado (matriz simétrica)
            K_G[i, j] = k_g_ij
            if i != j:
                K_G[j, i] = k_g_ij

    stop = time()
    print('Tempo de cálculo de K_G = ', stop - start)

    return K_G