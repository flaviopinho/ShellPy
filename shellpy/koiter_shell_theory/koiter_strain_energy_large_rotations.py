from time import time
import numpy as np

from shellpy import boole_weights_double_integral
from shellpy import Shell
from .koiter_strain_tensor import koiter_linear_strain_components, koiter_nonlinear_strain_components_total, \
    koiter_nonlinear_strain_components_total2


def koiter_strain_energy_large_rotations(shell: Shell, integral_weights=boole_weights_double_integral):
    """
    Calculates the strain energy functional for a shell structure using the Koiter approximation.
    This function computes quadratic, cubic, and quartic strain energy components.

    Parameters:
    - shell (Shell): The shell object containing all necessary properties, such as
      material properties, thickness, displacement expansions, and geometric data.
    - integral_weights (function): A function to calculate the integration weights and points for the domain.
      Defaults to 'boole_weights_double_integral'.

    Returns:
    - quadratic_energy_tensor (ndarray): The quadratic strain energy tensor.
    - cubic_energy_tensor (ndarray): The cubic strain energy tensor.
    - quartic_energy_tensor (ndarray): The quartic strain energy tensor.
    """

    # Get integration points and weights for the double integral over the mid-surface domain
    xi1, xi2, W = integral_weights(shell.mid_surface_domain.edges["xi1"], shell.mid_surface_domain.edges["xi2"])

    # Shape of xi1 (discretized domain in terms of xi1 and xi2)
    n = np.shape(xi1)

    # Number of degrees of freedom (dof) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Initialize arrays for linear strain components (gamma_lin) and their associated quantities (rho_lin)
    gamma_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))
    rho_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))

    # Compute the contravariant metric tensor components and sqrt(G) for the shell geometry
    G = shell.mid_surface_geometry.metric_tensor_contravariant_components(xi1, xi2)
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    # Calculate the constitutive tensor C for the thin shell material
    C = shell.material.plane_stress_constitutive_tensor_for_koiter_theory(G)

    # Get the thickness of the shell at each point in the domain
    h = shell.thickness(xi1, xi2)

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        gamma_lin[i], rho_lin[i] = koiter_linear_strain_components(shell.mid_surface_geometry,
                                                                   shell.displacement_expansion, i, xi1, xi2)
        print(f'Calculating linear components {i} of {n_dof}')

    # Initialize array for nonlinear strain components
    gamma_nonlin = np.zeros((n_dof, n_dof, 2, 2, n[0], n[1]))
    rho_nonlin = np.zeros((n_dof, n_dof, 2, 2, n[0], n[1]))

    # Loop through the degrees of freedom to compute nonlinear strain components
    for i in range(n_dof):
        for j in range(n_dof):
            gamma_nonlin[i, j], rho_nonlin[i, j] = koiter_nonlinear_strain_components_total2(shell.mid_surface_geometry,
                                                                                            shell.displacement_expansion,
                                                                                            i, j, xi1, xi2)
            print(f'Calculating nonlinear components ({i}, {j}) of ({n_dof}, {n_dof})')

    # Calculate the quadratic strain energy functional
    print('Calculating quadratic strain energy functional...')
    start = time()
    quadratic_energy_tensor = (h / 2) * np.einsum('abolxy, mabxy, nolxy, xy, xy->mn', C, gamma_lin, gamma_lin, sqrtG, W,
                                                  optimize=True)
    quadratic_energy_tensor += (h ** 3 / 24) * np.einsum('abolxy, mabxy, nolxy, xy, xy->mn', C, rho_lin, rho_lin, sqrtG,
                                                         W, optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Calculate the cubic strain energy functional
    print('Calculating cubic strain energy functional...')
    start = time()
    cubic_energy_tensor = 2 * (h / 2) * np.einsum('abcdxy, mabxy, nocdxy, xy, xy->mno', C, gamma_lin, gamma_nonlin,
                                                  sqrtG, W, optimize=True)

    cubic_energy_tensor += 2 * (h ** 3 / 24) * np.einsum('abcdxy, mabxy, nocdxy, xy, xy->mno', C, rho_lin, rho_nonlin,
                                                         sqrtG, W, optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Calculate the quartic strain energy functional
    print('Calculating quartic strain energy functional...')
    start = time()
    quartic_energy_tensor = (h / 2) * np.einsum('abcdxy, mnabxy, opcdxy, xy, xy->mnop', C, gamma_nonlin, gamma_nonlin,
                                                sqrtG, W, optimize=True)

    quartic_energy_tensor += (h ** 3 / 24) * np.einsum('abcdxy, mnabxy, opcdxy, xy, xy->mnop', C, rho_nonlin,
                                                      rho_nonlin,
                                                      sqrtG, W, optimize=True)

    stop = time()
    print('time= ', stop - start)

    # Return the computed strain energy tensors for quadratic, cubic, and quartic terms
    return quadratic_energy_tensor, cubic_energy_tensor, quartic_energy_tensor

