import numpy as np
from shellpy import Shell, cache_function
from shellpy.numeric_integration.integral_weights import double_integral_weights

# Adjust the imports below according to your folder structure
from .plane_stress_constitutive_matrix_in_material_frame import constitutive_matrix_in_material_frame
from .plane_stress_constitutive_matrix_in_shell_frame import plane_stress_constitutive_matrix_in_shell_frame
from .strain_vector import linear_sanders_koiter_strain_vector, nonlinear_koiter_strain_components_quadratic_vector


@cache_function
def compute_constant_shell_matrices(shell: Shell, n_x, n_y, n_z, integral_method):
    """
    Computes all shell matrices and tensors for the Sanders-Koiter formulation
    that are strictly independent of the displacement vector (u).

    Due to the @cache_function decorator, this computationally expensive "Offline Phase"
    runs only once per mesh setup, significantly speeding up non-linear solver iterations.
    """
    # --------------------------------------------------------------------------------
    # 1. GEOMETRY AND NUMERICAL INTEGRATION
    # --------------------------------------------------------------------------------
    # Obtain integration points and weights for the mid-surface area
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)

    # Determine thickness and through-the-thickness integration points
    h = shell.thickness(xi1, xi2)
    xi3, Wz = integral_method((-h / 2, h / 2), n_z)

    n_xy = np.shape(xi1)
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    # Geometric Jacobians and Shifter Tensor (accounts for parallel surface scaling)
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    det_shifter = shell.mid_surface_geometry.determinant_shifter_tensor(xi1, xi2, xi3)

    # Effective differential area weight
    Wxy1 = sqrtG * Wxy

    # --------------------------------------------------------------------------------
    # 2. CONSTITUTIVE MATRICES (THICKNESS INTEGRATED)
    # --------------------------------------------------------------------------------
    # Evaluate the material constitutive matrix and expand dimensions if it is homogeneous
    C_mat = constitutive_matrix_in_material_frame(shell.mid_surface_geometry, shell.material, (xi1, xi2, xi3))
    if C_mat.ndim == 2:
        C_mat = np.einsum('ij, xyz->ijxyz', C_mat, xi3 ** 0)

    # Transform to the curvilinear shell coordinate system
    C = plane_stress_constitutive_matrix_in_shell_frame(shell.mid_surface_geometry, C_mat, (xi1, xi2, xi3))

    # Integrate through the thickness to obtain the shell section stiffnesses:
    # C0: Membrane, C1: Coupling, C2: Bending
    C0 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 0 * det_shifter * Wz, optimize=True)
    C1 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 1 * det_shifter * Wz, optimize=True)
    C2 = np.einsum('ijxyz, xyz->ijxy', C, xi3 ** 2 * det_shifter * Wz, optimize=True)

    # --------------------------------------------------------------------------------
    # 3. KINEMATIC COMPONENTS (STRAIN TENSORS)
    # --------------------------------------------------------------------------------
    # Pre-allocate arrays for linear strains (eps_lin) and quadratic non-linear strains (eps_nl)
    # Dimensions are structured to map DOFs to integration points in Voigt notation (3 components)
    eps0_lin = np.zeros((n_dof, 3) + n_xy)
    eps1_lin = np.zeros((n_dof, 3) + n_xy)

    # Note: These quadratic tensors scale with O(N^2) memory footprint
    eps0_nl = np.zeros((n_dof, n_dof, 3) + n_xy)
    eps1_nl = np.zeros((n_dof, n_dof, 3) + n_xy)

    for i in range(n_dof):
        # Compute linear membrane and curvature strains for DOF 'i'
        eps0_lin[i], eps1_lin[i] = linear_sanders_koiter_strain_vector(
            shell.mid_surface_geometry, shell.displacement_expansion, i, xi1, xi2)

        for j in range(i, n_dof):
            # Compute quadratic coupled strain terms between DOFs 'i' and 'j'
            e0_ij, e1_ij = nonlinear_koiter_strain_components_quadratic_vector(
                shell.mid_surface_geometry, shell.displacement_expansion, i, j, xi1, xi2)

            eps0_nl[i, j] = e0_ij
            eps1_nl[i, j] = e1_ij

            # Enforce symmetry to safely populate the full DOF x DOF tensor
            if i != j:
                eps0_nl[j, i] = e0_ij
                eps1_nl[j, i] = e1_ij

    return Wxy1, C0, C1, C2, eps0_lin, eps1_lin, eps0_nl, eps1_nl