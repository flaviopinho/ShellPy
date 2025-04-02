from time import time
import numpy as np

from shellpy import Shell
from .koiter_strain_tensor import koiter_linear_strain_components
from .koiter_strain_tensor_large import koiter_nonlinear_strain_components_quadratic
from ..materials.constitutive_tensor_koiter import plane_stress_constitutive_tensor_for_koiter_theory
from ..numeric_integration.boole_integral import boole_weights_simple_integral
from ..numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_z, \
    n_integral_default_y
from ..numeric_integration.integral_weights import double_integral_weights


def koiter_strain_energy_large_rotations(shell: Shell,
                                         n_x=n_integral_default_x,
                                         n_y=n_integral_default_y, n_z=n_integral_default_z,
                                         integral_method=boole_weights_simple_integral):
    """
    Computes the strain energy functional for a shell structure using the Koiter approximation.
    This includes quadratic, cubic, and quartic strain energy components.

    Parameters:
    - shell (Shell): The shell object containing material properties, thickness, displacement expansions, and geometric data.
    - n_x, n_y, n_z (int): Number of integration points along each coordinate direction.
    - integral_method (function): Integration method used for numerical integration.

    Returns:
    - quadratic_energy_tensor (ndarray): Quadratic strain energy tensor.
    - cubic_energy_tensor (ndarray): Cubic strain energy tensor.
    - quartic_energy_tensor (ndarray): Quartic strain energy tensor.
    """
    # Compute integration points and weights for the mid-surface domain
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)

    # Compute thickness and integration points along the thickness direction
    h = shell.thickness()
    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    # Get the number of degrees of freedom for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Compute metric tensor and determinant-related quantities
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    # Compute the constitutive tensor for plane stress conditions
    C = plane_stress_constitutive_tensor_for_koiter_theory(shell.mid_surface_geometry, shell.material, xi1, xi2, xi3)

    # Precompute integration of the constitutive tensor along the thickness direction
    C0 = 0.5 * np.einsum('ijklxyz, z, xyz, z->ijklxy', C, xi3 ** 0, det_shifter_tensor, Wz)
    C1 = 0.5 * np.einsum('ijklxyz, z, xyz, z->ijklxy', C, xi3 ** 1, det_shifter_tensor, Wz)
    C2 = 0.5 * np.einsum('ijklxyz, z, xyz, z->ijklxy', C, xi3 ** 2, det_shifter_tensor, Wz)

    Wxy = sqrtG * Wxy  # Adjust integration weights for the mid-surface

    # Initialize arrays for linear strain components
    epsilon0_lin = np.zeros((n_dof, 2, 2, *xi1.shape))
    epsilon1_lin = np.zeros((n_dof, 2, 2, *xi1.shape))

    L0_lin = np.zeros_like(epsilon0_lin)
    L1_lin = np.zeros_like(epsilon1_lin)

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i] = koiter_linear_strain_components(shell.mid_surface_geometry,
                                                                           shell.displacement_expansion, i, xi1, xi2)

        L0_lin[i] = np.einsum('abcdxy,cdxy->abxy', C0, epsilon0_lin[i]) + np.einsum('abcdxy,cdxy->abxy', C1,
                                                                                    epsilon1_lin[i])
        L1_lin[i] = np.einsum('abcdxy,cdxy->abxy', C1, epsilon0_lin[i]) + np.einsum('abcdxy,cdxy->abxy', C2,
                                                                                    epsilon1_lin[i])

        print(f'Calculating linear components {i} of {n_dof}')

    # Initialize array for nonlinear strain components
    epsilon0_quadratic = np.zeros((n_dof, n_dof, 2, 2, *xi1.shape))
    epsilon1_quadratic = np.zeros((n_dof, n_dof, 2, 2, *xi1.shape))
    L0_quadratic = np.zeros_like(epsilon0_quadratic)
    L1_quadratic = np.zeros_like(epsilon0_quadratic)

    # Loop through the degrees of freedom to compute nonlinear strain components
    for i in range(n_dof):
        for j in range(i, n_dof):  # Compute only for i <= j to exploit symmetry
            gamma_ij, rho_ij = koiter_nonlinear_strain_components_quadratic(shell.mid_surface_geometry,
                                                                            shell.displacement_expansion,
                                                                            i, j, xi1, xi2)
            epsilon0_quadratic[i, j] = gamma_ij
            epsilon0_quadratic[j, i] = gamma_ij  # Exploit symmetry

            epsilon1_quadratic[i, j] = rho_ij
            epsilon1_quadratic[j, i] = rho_ij  # Exploit symmetry

            L0_quadratic[i, j] = np.einsum('abcdxy,cdxy->abxy', C0, epsilon0_quadratic[i, j]) + np.einsum(
                'abcdxy,cdxy->abxy', C1,
                epsilon1_quadratic[i, j])
            L1_quadratic[i, j] = np.einsum('abcdxy,cdxy->abxy', C1, epsilon0_quadratic[i, j]) + np.einsum(
                'abcdxy,cdxy->abxy', C2,
                epsilon1_quadratic[i, j])
            L0_quadratic[j, i] = L0_quadratic[i, j]
            L1_quadratic[j, i] = L1_quadratic[i, j]

            print(f'Calculating nonlinear components ({i}, {j}) of ({n_dof}, {n_dof})')


    # Calculate the quadratic strain energy functional
    print('Calculating quadratic strain energy functional...')
    start = time()
    quadratic_energy_tensor = np.einsum('mabxy, nabxy, xy->mn', L0_lin, epsilon0_lin, Wxy, optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', L1_lin, epsilon1_lin, Wxy, optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Calculate the cubic strain energy functional
    print('Calculating cubic strain energy functional...')
    start = time()
    cubic_energy_tensor = 2 * np.einsum('mabxy, noabxy, xy->mno', L0_lin, epsilon0_quadratic, Wxy, optimize=True)
    cubic_energy_tensor += 2 * np.einsum('mabxy, noabxy, xy->mno', L1_lin, epsilon0_quadratic, Wxy, optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Calculate the quartic strain energy functional
    print('Calculating quartic strain energy functional...')
    start = time()
    quartic_energy_tensor = np.einsum('mnabxy, opabxy, xy->mnop', L0_quadratic, epsilon0_quadratic, Wxy, optimize=True)
    quartic_energy_tensor += np.einsum('mnabxy, opabxy, xy->mnop', L1_quadratic, epsilon1_quadratic, Wxy, optimize=True)

    stop = time()
    print('time= ', stop - start)

    # Return the computed strain energy tensors for quadratic, cubic, and quartic terms
    return quadratic_energy_tensor, cubic_energy_tensor, quartic_energy_tensor  # , quint_energy_tensor, six_energy_tensor
