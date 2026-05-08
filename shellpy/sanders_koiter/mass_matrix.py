from time import time
import numpy as np
from ..shell import Shell
from ..displacement_covariant_derivative import displacement_first_covariant_derivatives
from ..materials.shell_density import shell_density
from ..numeric_integration.gauss_integral import gauss_weights_simple_integral
from ..numeric_integration.integral_weights import double_integral_weights


def mass_matrix(shell: Shell, n_x=20, n_y=20, n_z=10, integral_method=gauss_weights_simple_integral):
    # Get integration points and weights for the double integral over the mid-surface domain
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)

    h = shell.thickness(xi1, xi2)

    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    # Shape of xi1 (discretized domain in terms of xi1 and xi2)
    n_xy = np.shape(xi1)

    # Number of degrees of freedom (dof) for the displacement expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Compute the contravariant metric tensor components and sqrt(G) for the shell geometry
    metric_tensor_contravariant_components = shell.mid_surface_geometry.metric_tensor_contravariant_components_extended(
        xi1, xi2)

    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    Wxy1 = sqrtG * Wxy

    # Retrieve the density of the shell material at each integration point
    rho = shell_density(shell.material, xi1, xi2, xi3)

    # Compute weighted integral coefficients (moments of inertia of the cross-section)
    # The 1/2 factor is used because we will symmetrize the matrix at the end (M + M.T)
    W0 = 1 / 2 * np.einsum('xyz, xy->xy', rho * xi3 ** 0 * det_shifter_tensor * Wz, Wxy1)
    W1 = 1 / 2 * np.einsum('xyz, xy->xy', rho * xi3 ** 1 * det_shifter_tensor * Wz, Wxy1)
    W2 = 1 / 2 * np.einsum('xyz, xy->xy', rho * xi3 ** 2 * det_shifter_tensor * Wz, Wxy1)

    # Initialize arrays for displacements (u) and derived rotation/moment vectors (mu)
    u = np.zeros((n_dof, 3) + n_xy)
    mu = np.zeros((n_dof, 3) + n_xy)

    # Loop through the degrees of freedom to compute kinematic fields
    for i in range(n_dof):
        U = shell.displacement_expansion.shape_function(i, xi1, xi2)
        dU = shell.displacement_expansion.shape_function_first_derivatives(i, xi1, xi2)

        # In classical theory, we only extract the 3 mid-surface translations
        u_i = U[0:3]
        du_i = dU[0:3]

        u[i] = u_i

        # Compute the covariant derivatives of the displacement field
        dcu = displacement_first_covariant_derivatives(
            shell.mid_surface_geometry, u_i, du_i, xi1, xi2
        )

        # Convert covariant derivatives to contravariant form using the metric tensor
        dcu_contra = np.einsum('mi...,i...->m...', metric_tensor_contravariant_components, dcu)

        # Compute components of the linearized rotation vector mu
        mu[i, 0] = -dcu_contra[2, 0]  # -u^3_{|1}
        mu[i, 1] = -dcu_contra[2, 1]  # -u^3_{|2}
        mu[i, 2] = dcu_contra[0, 0] + dcu_contra[1, 1]  # u^1_{|1} + u^2_{|2}

        print(f'Calculating mass matrix components {i + 1} of {n_dof}', end='\r')
    print()

    print('Calculating quadratic kinetic energy functional (Mass Matrix)...')
    start = time()

    # First term: Inertia from purely translational displacement
    mass_matrix_tensor = np.einsum('ijxy, aixy, bjxy, xy->ab',
                                   metric_tensor_contravariant_components,
                                   u, u, W0, optimize=True)

    # Second term: Coupled translational-rotational inertia (first moment of area)
    # The factor 2 arises because (a+b)^2 expands to a^2 + 2ab + b^2
    mass_matrix_tensor += 2 * np.einsum('ijxy, aixy, bjxy, xy->ab',
                                        metric_tensor_contravariant_components,
                                        u, mu, W1, optimize=True)

    # Third term: Rotational inertia (second moment of area)
    mass_matrix_tensor += np.einsum('ijxy, aixy, bjxy, xy->ab',
                                    metric_tensor_contravariant_components,
                                    mu, mu, W2, optimize=True)
    stop = time()
    print('Assembly time = ', stop - start)

    # Enforce exact symmetry and absorb the 1/2 factor from the W integrals
    return mass_matrix_tensor + mass_matrix_tensor.T