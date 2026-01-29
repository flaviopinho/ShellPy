from time import time
import numpy as np

from shellpy import Shell
from .koiter_strain_tensor import koiter_linear_strain_components, koiter_nonlinear_strain_components_total
from shellpy.sanders_koiter.constitutive_tensor_koiter import plane_stress_constitutive_tensor_for_koiter_theory
from ..numeric_integration.boole_integral import boole_weights_simple_integral
from ..numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_z, \
    n_integral_default_y
from ..numeric_integration.gauss_integral import gauss_weights_simple_integral
from ..numeric_integration.integral_weights import double_integral_weights


def fast_koiter_strain_energy(shell: Shell,
                              n_x=n_integral_default_x,
                              n_y=n_integral_default_y,
                              n_z=n_integral_default_z,
                              integral_method=gauss_weights_simple_integral):
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
    h = shell.thickness(xi1, xi2)
    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    # Get the number of degrees of freedom for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Compute metric tensor and determinant-related quantities
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    # Compute the constitutive tensor for plane stress conditions
    C = plane_stress_constitutive_tensor_for_koiter_theory(shell.mid_surface_geometry, shell.material, xi1, xi2, xi3)

    # Precompute integration of the constitutive tensor along the thickness direction
    C0 = 0.5 * np.einsum('ijklxyz, xyz, xyz, xyz->ijklxy', C, xi3 ** 0, det_shifter_tensor, Wz)
    C1 = 0.5 * np.einsum('ijklxyz, xyz, xyz, xyz->ijklxy', C, xi3 ** 1, det_shifter_tensor, Wz)
    C2 = 0.5 * np.einsum('ijklxyz, xyz, xyz, xyz->ijklxy', C, xi3 ** 2, det_shifter_tensor, Wz)

    Wxy = sqrtG * Wxy  # Adjust integration weights for the mid-surface

    # Initialize arrays for linear strain components
    epsilon0_lin = np.zeros((n_dof, 2, 2, *xi1.shape))
    epsilon1_lin = np.zeros((n_dof, 2, 2, *xi1.shape))

    L0_lin = np.zeros_like(epsilon0_lin)
    L1_lin = np.zeros_like(epsilon1_lin)

    # Compute linear strain components for each degree of freedom
    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i] = koiter_linear_strain_components(
            shell.mid_surface_geometry, shell.displacement_expansion, i, xi1, xi2)

        L0_lin[i] = np.einsum('abcdxy,cdxy->abxy', C0, epsilon0_lin[i]) + np.einsum('abcdxy,cdxy->abxy', C1,
                                                                                    epsilon1_lin[i])
        L1_lin[i] = np.einsum('abcdxy,cdxy->abxy', C1, epsilon0_lin[i]) + np.einsum('abcdxy,cdxy->abxy', C2,
                                                                                    epsilon1_lin[i])

    # Initialize arrays for nonlinear strain components
    epsilon0_nonlin = np.zeros((n_dof, n_dof, 2, 2, *xi1.shape))
    L0_nonlin = np.zeros_like(epsilon0_nonlin)

    # Compute nonlinear strain components
    for i in range(n_dof):
        for j in range(i, n_dof):
            gamma_ij = koiter_nonlinear_strain_components_total(
                shell.mid_surface_geometry, shell.displacement_expansion, i, j, xi1, xi2)
            epsilon0_nonlin[i, j] = gamma_ij
            epsilon0_nonlin[j, i] = gamma_ij  # Exploit symmetry

            L0_nonlin[i, j] = np.einsum('abcdxy,cdxy->abxy', C0, epsilon0_nonlin[i, j])
            L0_nonlin[j, i] = L0_nonlin[i, j]

    # Compute strain energy functionals
    quadratic_energy_tensor = np.einsum('mabxy, nabxy, xy->mn', L0_lin, epsilon0_lin, Wxy, optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', L1_lin, epsilon1_lin, Wxy, optimize=True)

    cubic_energy_tensor = 2 * np.einsum('mabxy, noabxy, xy->mno', L0_lin, epsilon0_nonlin, Wxy, optimize=True)

    quartic_energy_tensor = np.einsum('mnabxy, opabxy, xy->mnop', L0_nonlin, epsilon0_nonlin, Wxy, optimize=True)

    return quadratic_energy_tensor, cubic_energy_tensor, quartic_energy_tensor


def fast_koiter_quadratic_strain_energy(shell: Shell,
                                        n_x=n_integral_default_x,
                                        n_y=n_integral_default_y, n_z=n_integral_default_z,
                                        integral_method=boole_weights_simple_integral):
    """
    Computes the strain energy functional for a shell structure using the Koiter approximation.
    This includes only the quadratic strain energy component.

    Parameters:
    - shell (Shell): The shell object containing material properties, thickness, displacement expansions, and geometric data.
    - n_x, n_y, n_z (int): Number of integration points along each coordinate direction.
    - integral_method (function): Integration method used for numerical integration.

    Returns:
    - quadratic_energy_tensor (ndarray): Quadratic strain energy tensor.
    """

    # Get integration points and weights for the double integral over the mid-surface domain
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)

    # Get the thickness of the shell at each point in the domain
    h = shell.thickness(xi1, xi2)

    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    # Shape of xi1 (discretized domain in terms of xi1 and xi2)
    n = np.shape(xi1)

    # Number of degrees of freedom (dof) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Compute the contravariant metric tensor components and sqrt(G) for the shell geometry
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    # Calculate the constitutive tensor C for the thin shell material
    C = plane_stress_constitutive_tensor_for_koiter_theory(shell.mid_surface_geometry, shell.material, xi1, xi2, xi3)

    C0 = 1 / 2 * np.einsum('ijklxyz, xyz, xyz, xyz->ijklxy', C, xi3 ** 0, det_shifter_tensor, Wz)
    C1 = 1 / 2 * np.einsum('ijklxyz, xyz, xyz, xyz->ijklxy', C, xi3 ** 1, det_shifter_tensor, Wz)
    C2 = 1 / 2 * np.einsum('ijklxyz, xyz, xyz, xyz->ijklxy', C, xi3 ** 2, det_shifter_tensor, Wz)

    Wxy = sqrtG * Wxy

    # Initialize arrays for linear strain components (gamma_lin) and their associated quantities (rho_lin)
    epsilon0_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))
    epsilon1_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))

    L0_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))
    L1_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i] = koiter_linear_strain_components(shell.mid_surface_geometry,
                                                                           shell.displacement_expansion,
                                                                           i, xi1, xi2)

        L0_lin[i] = (np.einsum('abcdxy,cdxy->abxy', C0, epsilon0_lin[i]) +
                     np.einsum('abcdxy,cdxy->abxy', C1, epsilon1_lin[i]))
        L1_lin[i] = (np.einsum('abcdxy,cdxy->abxy', C1, epsilon0_lin[i]) +
                     np.einsum('abcdxy,cdxy->abxy', C2, epsilon1_lin[i]))

        print(f'Calculating linear components {i} of {n_dof}')

    # Calculate the quadratic strain energy functional
    print('Calculating quadratic strain energy functional...')
    start = time()
    quadratic_energy_tensor = np.einsum('mabxy, nabxy, xy->mn',
                                        L0_lin, epsilon0_lin, Wxy, optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn',
                                         L1_lin, epsilon1_lin, Wxy, optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Return the computed strain energy tensors for quadratic, cubic, and quartic terms
    return quadratic_energy_tensor
