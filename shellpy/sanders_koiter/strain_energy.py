from time import time
import numpy as np

from .plane_stress_constitutive_matrix_in_material_frame import constitutive_matrix_in_material_frame
from .plane_stress_constitutive_matrix_in_shell_frame import plane_stress_constitutive_matrix_in_shell_frame
from .strain_vector import linear_sanders_koiter_strain_vector, nonlinear_koiter_strain_components_quadratic_vector
from ..shell import Shell
from ..numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_z, \
    n_integral_default_y
from ..numeric_integration.gauss_integral import gauss_weights_simple_integral
from ..numeric_integration.integral_weights import double_integral_weights


def strain_energy(shell: Shell,
                  n_x=n_integral_default_x,
                  n_y=n_integral_default_y,
                  n_z=n_integral_default_z,
                  integral_method=gauss_weights_simple_integral):
    """
    Computes the strain energy functional (Quadratic, Cubic, Quartic) using Koiter Theory
    with vectorized Voigt notation (3x3 constitutive matrices).
    """
    # 1. Integração Numérica e Pesos
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)
    h = shell.thickness(xi1, xi2)
    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    n_xy = np.shape(xi1)
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    det_shifter = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)
    Wxy1 = sqrtG * Wxy

    # 2. Tensor Constitutivo (Convertido para matriz 3x3 de Voigt para Koiter)
    # Nota: plane_stress_constitutive_tensor_for_koiter_theory deve retornar/ser adaptada para 3x3xy
    # Calculate the constitutive tensor C for the thin shell material
    C_material = constitutive_matrix_in_material_frame(shell.mid_surface_geometry, shell.material, (xi1, xi2, xi3))
    C = plane_stress_constitutive_matrix_in_shell_frame(shell.mid_surface_geometry, C_material, (xi1, xi2, xi3))

    # Integração na espessura para obter as rigidezes de membrana, acoplamento e flexão
    # Formato resultante: (3, 3, nx, ny)
    C0 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 0 * det_shifter * Wz)
    C1 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 1 * det_shifter * Wz)
    C2 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 2 * det_shifter * Wz)

    # 3. Componentes Lineares (Vetorizados Voigt 3x1)
    epsilon0_lin = np.zeros((n_dof, 3) + n_xy)
    epsilon1_lin = np.zeros((n_dof, 3) + n_xy)
    L0_lin = np.zeros_like(epsilon0_lin)
    L1_lin = np.zeros_like(epsilon1_lin)

    for i in range(n_dof):
        # Retorna vetores 3x1 (e11, e22, g12)
        epsilon0_lin[i], epsilon1_lin[i] = linear_sanders_koiter_strain_vector(
            shell.mid_surface_geometry, shell.displacement_expansion, i, xi1, xi2)

        L0_lin[i] = (np.einsum('ijxy, jxy->ixy', C0, epsilon0_lin[i], optimize=True) +
                     np.einsum('ijxy, jxy->ixy', C1, epsilon1_lin[i], optimize=True))

        L1_lin[i] = (np.einsum('ijxy, jxy->ixy', C1, epsilon0_lin[i], optimize=True) +
                     np.einsum('ijxy, jxy->ixy', C2, epsilon1_lin[i], optimize=True))

        if (i + 1) % 10 == 0 or i == n_dof - 1:
            print(f'Linear components: {i + 1}/{n_dof}')

    # 4. Componentes Não Lineares (Quadráticos em relação aos DOFs)
    epsilon0_nl = np.zeros((n_dof, n_dof, 3) + n_xy)
    epsilon1_nl = np.zeros((n_dof, n_dof, 3) + n_xy)
    L0_nl = np.zeros_like(epsilon0_nl)
    L1_nl = np.zeros_like(epsilon1_nl)

    for i in range(n_dof):
        for j in range(i, n_dof):
            # Retorna a contribuição quadrática para o par (i, j)
            eps0_ij, eps1_ij = nonlinear_koiter_strain_components_quadratic_vector(
                shell.mid_surface_geometry, shell.displacement_expansion, i, j, xi1, xi2)

            epsilon0_nl[i, j] = eps0_ij
            epsilon1_nl[i, j] = eps1_ij

            # Forças generalizadas não lineares
            L0_nl[i, j] = (np.einsum('ijxy, jxy->ixy', C0, eps0_ij, optimize=True) +
                           np.einsum('ijxy, jxy->ixy', C1, eps1_ij, optimize=True))
            L1_nl[i, j] = (np.einsum('ijxy, jxy->ixy', C1, eps0_ij, optimize=True) +
                           np.einsum('ijxy, jxy->ixy', C2, eps1_ij, optimize=True))

            # Simetria
            if i != j:
                epsilon0_nl[j, i] = epsilon0_nl[i, j]
                epsilon1_nl[j, i] = epsilon1_nl[i, j]
                L0_nl[j, i] = L0_nl[i, j]
                L1_nl[j, i] = L1_nl[i, j]

        print(f'Nonlinear components: row {i + 1}/{n_dof}')

    # 5. Cálculo das Energias (Tensores de Rigidez Global)
    print('Computing global energy tensors...')

    # Quadrática (K_mn) - 0.5 * integral(L_lin * eps_lin)
    start = time()
    quadratic_energy = 0.5 * (np.einsum('maxy, naxy, xy->mn', L0_lin, epsilon0_lin, Wxy1, optimize=True) +
                              np.einsum('maxy, naxy, xy->mn', L1_lin, epsilon1_lin, Wxy1, optimize=True))
    print(f'Quadratic energy time: {time() - start:.4f}s')

    # Cúbica (K_mno) - integral(L_lin * eps_nl) -> Nota: fator 2 ou 0.5 depende da convenção da expansão
    start = time()
    cubic_energy = (np.einsum('maxy, noaxy, xy->mno', L0_lin, epsilon0_nl, Wxy1, optimize=True) +
                    np.einsum('maxy, noaxy, xy->mno', L1_lin, epsilon1_nl, Wxy1, optimize=True))
    print(f'Cubic energy time: {time() - start:.4f}s')

    # Quártica (K_mnop) - 0.5 * integral(L_nl * eps_nl)
    start = time()
    quartic_energy = 0.5 * (np.einsum('mnaxy, opaxy, xy->mnop', L0_nl, epsilon0_nl, Wxy1, optimize=True) +
                            np.einsum('mnaxy, opaxy, xy->mnop', L1_nl, epsilon1_nl, Wxy1, optimize=True))
    print(f'Quartic energy time: {time() - start:.4f}s')

    return quadratic_energy, cubic_energy, quartic_energy
