from time import time
import numpy as np

from .plane_stress_constitutive_matrix_in_shell_frame import plane_stress_constitutive_matrix_in_shell_frame
from .plane_stress_constitutive_matrix_in_material_frame import constitutive_matrix_in_material_frame
from ..shell import Shell
from ..numeric_integration.gauss_integral import gauss_weights_simple_integral
from ..numeric_integration.integral_weights import double_integral_weights

# Importing the 3x3 linear strain functions validated for Voigt notation
from .strain_vector import linear_sanders_koiter_strain_vector


def stiffness_matrix_sanders_koiter(shell: Shell, n_x=20, n_y=20, n_z=10,
                                    integral_method=gauss_weights_simple_integral):
    """
    Computes the global linear stiffness matrix of the shell using Sanders-Koiter theory.

    This function performs numerical integration over the mid-surface domain and
    through the thickness to assemble the quadratic strain energy functional.

    Parameters
    ----------
    shell : Shell
        The shell object containing geometry, material, and displacement expansion.
    n_x, n_y : int
        Number of integration points in the xi1 and xi2 directions.
    n_z : int
        Number of integration points through the thickness (xi3).
    integral_method : function
        Numerical integration scheme (default is Gauss-Legendre weights).

    Returns
    -------
    stiffness_matrix : ndarray
        The assembled (n_dof x n_dof) symmetric linear stiffness matrix.
    """

    # --- 1. Domain Discretization and Integration Weights ---
    # Obtain integration points and weights for the double integral over the mid-surface domain
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)

    # Determine shell thickness at integration points and set up thickness integration (xi3)
    h = shell.thickness(xi1, xi2)
    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    n_xy = np.shape(xi1)

    # Total number of generalized coordinates (DOF) for the Ritz expansion
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # --- 2. Geometric Coefficients ---
    # sqrtG: Jacobian determinant of the mid-surface mapping
    # det_shifter_tensor: Accounts for the parallel surface metric scaling through the thickness
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    det_shifter_tensor = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    # Effective differential area weight
    Wxy1 = sqrtG * Wxy

    # --- 3. Constitutive Law Assembly ---
    # Compute the 3x3 plane-stress constitutive tensor in the material reference frame
    C_material = constitutive_matrix_in_material_frame(shell.mid_surface_geometry, shell.material, (xi1, xi2, xi3))

    # Expand dimensions for homogeneous isotropic materials if necessary
    if C_material.ndim == 2:
        C_material = np.einsum('ij, xyz->ijxyz', C_material, xi3 ** 0)

    # Transform the constitutive law to the reciprocal curvilinear shell basis
    C = plane_stress_constitutive_matrix_in_shell_frame(shell.mid_surface_geometry, C_material, (xi1, xi2, xi3))

    # Perform thickness integration to obtain Integrated Stiffness Tensors:
    # C0: Membrane, C1: Coupling, C2: Bending
    C0 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 0 * det_shifter_tensor * Wz, optimize=True)
    C1 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 1 * det_shifter_tensor * Wz, optimize=True)
    C2 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 2 * det_shifter_tensor * Wz, optimize=True)

    # --- 4. Linear Kinematics and Stress Resultants ---
    # Initialize arrays for linear strain components in Classic Voigt order [11, 22, 12]
    epsilon0_lin = np.zeros((n_dof, 3) + n_xy)  # Membrane strain (gamma)
    epsilon1_lin = np.zeros((n_dof, 3) + n_xy)  # Curvature change (rho)

    # Initialize arrays for integrated modal stress resultants N (L0) and M (L1)
    L0_lin = np.zeros((n_dof, 3) + n_xy)
    L1_lin = np.zeros((n_dof, 3) + n_xy)

    # Loop through basis functions (Ritz degrees of freedom)
    for i in range(n_dof):
        # Calculate modal strain components for DOF 'i'
        epsilon0_lin[i], epsilon1_lin[i] = linear_sanders_koiter_strain_vector(
            shell.mid_surface_geometry, shell.displacement_expansion, i, xi1, xi2
        )

        # Compute membrane force resultants (N = C0*eps0 + C1*eps1)
        L0_lin[i] = (np.einsum('ijxy, jxy->ixy', C0, epsilon0_lin[i], optimize=True) +
                     np.einsum('ijxy, jxy->ixy', C1, epsilon1_lin[i], optimize=True))

        # Compute bending moment resultants (M = C1*eps0 + C2*eps1)
        L1_lin[i] = (np.einsum('ijxy, jxy->ixy', C1, epsilon0_lin[i], optimize=True) +
                     np.einsum('ijxy, jxy->ixy', C2, epsilon1_lin[i], optimize=True))

        print(f'Calculating linear components {i + 1} of {n_dof}', end='\r')
    print()

    # --- 5. Global Stiffness Matrix Assembly ---
    print('Calculating quadratic strain energy functional (Stiffness Matrix)...')
    start = time()

    # Vectorized assembly: K = Integral( epsilon^T * Stress_Resultant ) dA
    # This efficiently computes the double summation over the domain area
    stiffness_matrix = np.einsum('maxy, naxy, xy->mn', L0_lin, epsilon0_lin, Wxy1, optimize=True)
    stiffness_matrix += np.einsum('maxy, naxy, xy->mn', L1_lin, epsilon1_lin, Wxy1, optimize=True)

    # Enforce numerical symmetry to eliminate minor floating-point residuals
    stiffness_matrix = (stiffness_matrix + stiffness_matrix.T) / 2

    stop = time()
    print('Assembly time = ', stop - start)

    return stiffness_matrix
