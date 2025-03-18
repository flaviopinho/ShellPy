from time import time
import numpy as np

from shellpy import boole_weights_double_integral, n_integral_default_x, n_integral_default_y
from shellpy import Shell
from .koiter_strain_tensor import koiter_linear_strain_components
from .koiter_strain_tensor_large import koiter_nonlinear_strain_components_quadratic, \
    koiter_nonlinear_strain_components_cubic


def koiter_strain_energy_large_rotations(shell: Shell, n_x=n_integral_default_x, n_y=n_integral_default_y, integral_weights=boole_weights_double_integral):
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

    W0 = (h / 2 * np.ones(np.shape(xi1)) + h ** 3 / 24 * gaussian_curvature) * sqrtG * W

    W1 = (h ** 3 / 24 * np.ones(np.shape(xi1)) + h ** 5 / 160 * gaussian_curvature) * sqrtG * W

    W2 = (h ** 3 / 24 * trace_K) * sqrtG * W

    # Initialize arrays for linear strain components (gamma_lin) and their associated quantities (rho_lin)
    epsilon0_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))
    epsilon1_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))

    L0_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))
    L1_lin = np.zeros((n_dof, 2, 2, n[0], n[1]))

    # Loop through the degrees of freedom to compute linear strain components for each dof
    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i] = koiter_linear_strain_components(shell.mid_surface_geometry,
                                                                   shell.displacement_expansion, i, xi1, xi2)

        L0_lin[i] = np.einsum('abcdxy,cdxy, xy->abxy', C, epsilon0_lin[i], W0)
        L0_lin[i] += np.einsum('abcdxy,cdxy, xy->abxy', C, epsilon1_lin[i], W1)

        L1_lin[i] = np.einsum('abcdxy,cdxy, xy->abxy', C, epsilon0_lin[i], W1)
        L1_lin[i] += np.einsum('abcdxy,cdxy, xy->abxy', C, epsilon1_lin[i], W2)

        print(f'Calculating linear components {i} of {n_dof}')

    # Initialize array for nonlinear strain components
    epsilon0_quadratic = np.zeros((n_dof, n_dof, 2, 2, n[0], n[1]))
    epsilon1_quadratic = np.zeros((n_dof, n_dof, 2, 2, n[0], n[1]))
    L0_quadratic = np.zeros((n_dof, n_dof, 2, 2, n[0], n[1]))
    L1_quadratic = np.zeros((n_dof, n_dof, 2, 2, n[0], n[1]))

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

            L0_quadratic[i, j] = np.einsum('abcdxy,cdxy,xy->abxy', C, epsilon0_quadratic[i, j], W0)
            L0_quadratic[i, j] += np.einsum('abcdxy,cdxy,xy->abxy', C, epsilon1_quadratic[i, j], W1)
            L0_quadratic[j, i] = L0_quadratic[i, j]

            L1_quadratic[i, j] = np.einsum('abcdxy,cdxy,xy->abxy', C, epsilon0_quadratic[i, j], W1)
            L1_quadratic[i, j] += np.einsum('abcdxy,cdxy,xy->abxy', C, epsilon1_quadratic[i, j], W2)
            L1_quadratic[j, i] = L1_quadratic[i, j]

            print(f'Calculating nonlinear components ({i}, {j}) of ({n_dof}, {n_dof})')

    """
    # Loop through the degrees of freedom to compute nonlinear strain components
    epsilon1_cubic = np.zeros((n_dof, n_dof, n_dof, 2, 2, n[0], n[1]))
    L1_cubic = np.zeros((n_dof, n_dof, n_dof, 2, 2, n[0], n[1]))
    for i in range(n_dof):
        for j in range(i, n_dof):
            for k in range(n_dof):
                epsilon1_cubic[i, j, k] = koiter_nonlinear_strain_components_cubic(shell.mid_surface_geometry,
                                                                             shell.displacement_expansion,
                                                                             i, j, k, xi1, xi2)

                L1_cubic[i, j, k] = np.einsum('abcdxy,cdxy,xy->abxy', C, epsilon1_cubic[i, j, k], W2)

                print(f'Calculating nonlinear components ({i}, {j}, {k}) of ({n_dof}, {n_dof}, {n_dof})')
    """

    # Calculate the quadratic strain energy functional
    print('Calculating quadratic strain energy functional...')
    start = time()
    quadratic_energy_tensor = np.einsum('mabxy, nabxy->mn', L0_lin, epsilon0_lin, optimize=True)
    quadratic_energy_tensor += np.einsum('mabxy, nabxy->mn', L1_lin, epsilon1_lin, optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Calculate the cubic strain energy functional
    print('Calculating cubic strain energy functional...')
    start = time()
    cubic_energy_tensor = 2 * np.einsum('mabxy, noabxy->mno', L0_lin, epsilon0_quadratic, optimize=True)
    cubic_energy_tensor += 2 * np.einsum('mabxy, noabxy->mno', L1_lin, epsilon0_quadratic, optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Calculate the quartic strain energy functional
    print('Calculating quartic strain energy functional...')
    start = time()
    quartic_energy_tensor = np.einsum('mnabxy, opabxy->mnop', L0_quadratic, epsilon0_quadratic, optimize=True)
    quartic_energy_tensor += np.einsum('mnabxy, opabxy->mnop', L1_quadratic, epsilon1_quadratic, optimize=True)

    #quartic_energy_tensor += 2 * np.einsum('mabxy, nopabxy->mnop', L1_lin, epsilon1_cubic, optimize=True)

    stop = time()
    print('time= ', stop - start)

    """
    # Calculate the quartic strain energy functional
    print('Calculating 5th strain energy functional...')
    start = time()
    quint_energy_tensor = 2 * np.einsum('mnabxy, opqabxy->mnopq', L1_quadratic, epsilon1_cubic, optimize=True)
    stop = time()
    print('time= ', stop - start)

    # Calculate the quartic strain energy functional
    print('Calculating 6th strain energy functional...')
    start = time()
    six_energy_tensor = np.einsum('mnoabxy, pqrabxy->mnopqr', L1_cubic, epsilon1_cubic, optimize=True)
    stop = time()
    print('time= ', stop - start) """

    # Return the computed strain energy tensors for quadratic, cubic, and quartic terms
    return quadratic_energy_tensor, cubic_energy_tensor, quartic_energy_tensor #, quint_energy_tensor, six_energy_tensor

