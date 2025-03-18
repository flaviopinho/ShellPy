from time import time
import numpy as np

from shellpy import boole_weights_double_integral, n_integral_default_x, n_integral_default_y
from shellpy import Shell
from .koiter_strain_tensor import koiter_linear_strain_components, koiter_nonlinear_strain_components_total


def fast_koiter_strain_energy(shell: Shell, n_x=n_integral_default_x, n_y=n_integral_default_y, integral_weights=boole_weights_double_integral):
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
    xi1, xi2, W = integral_weights(shell.mid_surface_domain.edges["xi1"], shell.mid_surface_domain.edges["xi2"], n_x, n_y)

    # Shape of xi1 (discretized domain in terms of xi1 and xi2)
    n = np.shape(xi1)

    # Number of degrees of freedom (dof) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Compute the contravariant metric tensor components and sqrt(G) for the shell geometry
    G = shell.mid_surface_geometry.metric_tensor_contravariant_components(xi1, xi2)
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    gaussian_curvature = shell.mid_surface_geometry.gaussian_curvature(xi1, xi2)

    trace_K = 2 * shell.mid_surface_geometry.mean_curvature(xi1, xi2)

    # Calculate the constitutive tensor C for the thin shell material
    C = shell.material.plane_stress_constitutive_tensor_for_koiter_theory(G)

    # Get the thickness of the shell at each point in the domain
    h = shell.thickness(xi1, xi2)

    W1 = (h / 2 * np.ones(np.shape(xi1)) + h ** 3 / 24 * gaussian_curvature) * sqrtG * W

    W2 = (h ** 3 / 24 * np.ones(np.shape(xi1)) + h ** 5 / 160 * gaussian_curvature) * sqrtG * W

    W3 = (h ** 3 / 24 * trace_K) * sqrtG * W

    # Initialize arrays for linear strain components (gamma_lin) and their associated quantities (rho_lin)
    epsilon0_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))
    epsilon1_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))

    sigma0_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))
    sigma1_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i] = koiter_linear_strain_components(shell.mid_surface_geometry,
                                                                   shell.displacement_expansion, i, xi1, xi2)

        sigma0_lin[i] = np.einsum('abcdxy,cdxy->abxy', C, epsilon0_lin[i])
        sigma1_lin[i] = np.einsum('abcdxy,cdxy->abxy', C, epsilon1_lin[i])

        print(f'Calculating linear components {i} of {n_dof}')

    # Initialize array for nonlinear strain components
    epsilon0_nonlin = np.zeros((n_dof, n_dof, 2, 2, n[0], n[1]))
    sigma0_nonlin = np.zeros((n_dof, n_dof, 2, 2, n[0], n[1]))

    # Loop through the degrees of freedom to compute nonlinear strain components
    for i in range(n_dof):
        for j in range(i, n_dof):  # Compute only for i <= j to exploit symmetry
            gamma_ij = koiter_nonlinear_strain_components_total(shell.mid_surface_geometry,
                                                                shell.displacement_expansion,
                                                                i, j, xi1, xi2)
            epsilon0_nonlin[i, j] = gamma_ij
            epsilon0_nonlin[j, i] = gamma_ij  # Exploit symmetry

            sigma0_nonlin[i, j] = np.einsum('abcdxy,cdxy->abxy', C, epsilon0_nonlin[i, j])
            sigma0_nonlin[j, i] = sigma0_nonlin[i, j]


            print(f'Calculating nonlinear components ({i}, {j}) of ({n_dof}, {n_dof})')

    # Calculate the quadratic strain energy functional
    print('Calculating quadratic strain energy functional...')
    start = time()
    quadratic_energy_tensor = np.einsum('mabxy, nabxy, xy->mn', sigma0_lin, epsilon0_lin, W1, optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', sigma1_lin, epsilon1_lin, W2, optimize=True)
    quadratic_energy_tensor += 2 * np.einsum('mabxy, nabxy, xy->mn', sigma0_lin, epsilon1_lin, W3, optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Calculate the cubic strain energy functional
    print('Calculating cubic strain energy functional...')
    start = time()
    cubic_energy_tensor = 2 * np.einsum('mabxy, noabxy, xy->mno', sigma0_lin, epsilon0_nonlin, W1, optimize=True)
    cubic_energy_tensor += 2 * np.einsum('mabxy, noabxy, xy->mno', sigma1_lin, epsilon0_nonlin, W3, optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Calculate the quartic strain energy functional
    print('Calculating quartic strain energy functional...')
    start = time()
    quartic_energy_tensor = np.einsum('mnabxy, opabxy, xy->mnop', sigma0_nonlin, epsilon0_nonlin, W1, optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Return the computed strain energy tensors for quadratic, cubic, and quartic terms
    return quadratic_energy_tensor, cubic_energy_tensor, quartic_energy_tensor


def fast_koiter_quadratic_strain_energy(shell: Shell, n_x=n_integral_default_x, n_y=n_integral_default_y, integral_weights=boole_weights_double_integral):
    """
    Calculates only the quadratic strain energy functional for a shell structure using the Koiter approximation.

    Parameters:
    - shell (Shell): The shell object containing all necessary properties, such as
      material properties, thickness, displacement expansions, and geometric data.
    - integral_weights (function): A function to calculate the integration weights and points for the domain.
      Defaults to 'boole_weights_double_integral'.

    Returns:
    - quadratic_energy_tensor (ndarray): The quadratic strain energy tensor.
    """

    # Get integration points and weights for the double integral over the mid-surface domain
    xi1, xi2, W = integral_weights(shell.mid_surface_domain.edges["xi1"], shell.mid_surface_domain.edges["xi2"], n_x, n_y)

    # Shape of xi1 (discretized domain in terms of xi1 and xi2)
    n = np.shape(xi1)

    # Number of degrees of freedom (dof) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Initialize arrays for linear strain components (gamma_lin) and their associated quantities (rho_lin)
    epsilon0_lin = np.zeros((n_dof, 2, 2) + n)
    epsilon1_lin = np.zeros((n_dof, 2, 2) + n)
    sigma0_lin = np.zeros((n_dof, 2, 2) + n)
    sigma1_lin = np.zeros((n_dof, 2, 2) + n)

    # Compute the contravariant metric tensor components and sqrt(G) for the shell geometry
    G = shell.mid_surface_geometry.metric_tensor_contravariant_components(xi1, xi2)
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    gaussian_curvature = shell.mid_surface_geometry.gaussian_curvature(xi1, xi2)

    trace_K = 2 * shell.mid_surface_geometry.mean_curvature(xi1, xi2)

    # Calculate the constitutive tensor C for the thin shell material
    C = shell.material.plane_stress_constitutive_tensor_for_koiter_theory(G)

    # Get the thickness of the shell at each point in the domain
    h = shell.thickness(xi1, xi2)

    W1 = (h / 2 * np.ones(np.shape(xi1)) + h ** 3 / 24 * gaussian_curvature) * sqrtG * W

    W2 = (h ** 3 / 24 * np.ones(np.shape(xi1)) + h ** 5 / 160 * gaussian_curvature) * sqrtG * W

    W3 = (h ** 3 / 24 * trace_K) * sqrtG * W

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i] = koiter_linear_strain_components(shell.mid_surface_geometry,
                                                                           shell.displacement_expansion, i, xi1, xi2)

        sigma0_lin[i] = np.einsum('abcdxy,cdxy->abxy', C, epsilon0_lin[i])
        sigma1_lin[i] = np.einsum('abcdxy,cdxy->abxy', C, epsilon1_lin[i])

        print(f'Calculating linear components {i} of {n_dof}')

    # Calculate the quadratic strain energy functional
    print('Calculating quadratic strain energy functional...')
    start = time()
    quadratic_energy_tensor = np.einsum('mabxy, nabxy, xy->mn', sigma0_lin, epsilon0_lin, W1, optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy, xy->mn', sigma1_lin, epsilon1_lin, W2, optimize=True)
    quadratic_energy_tensor += 2 * np.einsum('mabxy, nabxy, xy->mn', sigma0_lin, epsilon1_lin, W3, optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Return the quadratic strain energy tensor
    return quadratic_energy_tensor