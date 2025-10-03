import numpy as np
from multipledispatch import dispatch

from shellpy import MidSurfaceGeometry, cache_function
from shellpy.materials.functionally_graded_material import FunctionallyGradedMaterial
from shellpy.materials.laminate_orthotropic_material import LaminateOrthotropicMaterial
from shellpy.materials.linear_elastic_material import LinearElasticMaterial
from shellpy.materials.transformation_matrix_fosd2 import transformation_matrix_fosd2_global, \
    transformation_matrix_fosd2_alpha


@dispatch(MidSurfaceGeometry, LinearElasticMaterial, np.ndarray, np.ndarray, np.ndarray)
@cache_function
def constitutive_matrix_for_fosd2(mid_surface_geometry, material, xi1, xi2, xi3):
    """
    Compute the constitutive tensor in the shell reciprocal frame for FOSD2.

    Parameters
    ----------
    mid_surface_geometry : MidSurfaceGeometry
        Object containing the shell geometry and metric properties.
    material : LinearElasticMaterial
        Material object with properties E (Young's modulus) and nu (Poisson's ratio)
    xi1, xi2, xi3 : arrays
        Coordinates in the curvilinear parametric space.

    Returns
    -------
    C_shell_reciprocal_frame : array
        Constitutive tensor in Voigt notation in the shell reciprocal frame.
        Shape: (6,6, ...) matching the shape of xi1, xi2, xi3
    """

    T = transformation_matrix_fosd2_global(mid_surface_geometry, xi1, xi2, xi3)

    # -----------------------------
    # Constitutive matrix in the local material frame (isotropic case, Voigt 6x6)
    # -----------------------------
    E = material.E
    nu = material.nu
    factor = E / ((1 + nu) * (1 - 2 * nu))
    C_local_material = factor * np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
    ])

    # -----------------------------
    # Compute the constitutive tensor in the shell reciprocal frame
    # C_shell = T^T @ C_local @ T using einsum
    # -----------------------------
    C_shell_reciprocal_frame = np.einsum('ji...,jk,kl...->il...', T, C_local_material, T)

    return C_shell_reciprocal_frame


@dispatch(MidSurfaceGeometry, FunctionallyGradedMaterial, np.ndarray, np.ndarray, np.ndarray)
@cache_function
def constitutive_matrix_for_fosd2(mid_surface_geometry: MidSurfaceGeometry, material: FunctionallyGradedMaterial, xi1,
                                 xi2,
                                 xi3):
    """
        Compute the constitutive tensor in the shell reciprocal frame for FOSD2.

        Parameters
        ----------
        mid_surface_geometry : MidSurfaceGeometry
            Object containing the shell geometry and metric properties.
        material : FunctionallyGradedMaterial
            Material object with properties E (Young's modulus) and nu (Poisson's ratio)
            depending on xi^3
        xi1, xi2, xi3 : arrays
            Coordinates in the curvilinear parametric space.

        Returns
        -------
        C_shell_reciprocal_frame : array
            Constitutive tensor in Voigt notation in the shell reciprocal frame.
            Shape: (6,6, ...) matching the shape of xi1, xi2, xi3
        """

    # -----------------------------
    # Material directions in global frame
    # -----------------------------
    e1 = np.array((1, 0, 0))
    e2 = np.array((0, 1, 0))
    e3 = np.array((0, 0, 1))
    material_base = (e1, e2, e3)

    # -----------------------------
    # Obtain the reciprocal base from the shell geometry
    # -----------------------------
    reciprocal_base = mid_surface_geometry.reciprocal_base(xi1, xi2)

    # -----------------------------
    # Obtain the shifter tensor (2x2 in-plane approximation)
    # -----------------------------
    shifter_tensor_inverse = mid_surface_geometry.shifter_tensor_inverse_approximation(xi1, xi2, xi3)

    # -----------------------------
    # Convert tuples/lists to 3x3 arrays
    # -----------------------------
    MR = np.stack(reciprocal_base, axis=0)  # shape (3,3,...)
    e = np.stack(material_base, axis=0)  # shape (3,3)

    # -----------------------------
    # Compute the A tensor: A[i,j] = shifter @ M_block @ e_block
    # Only in-plane 2x2 block, 3rd component set to 1
    # -----------------------------
    Lambda = shifter_tensor_inverse  # shape (2,2,...)
    MR_block = MR[0:2, 0:2]  # shape (2,2,...)
    e_block = e[0:2, 0:2]  # shape (2,2)

    # Use einsum for tensor contraction over in-plane indices
    R_block = np.einsum('ik...,kl...,lj->ij...', Lambda, MR_block, e_block)

    # Assemble the full 3x3 A tensor
    R = np.zeros((3, 3) + R_block.shape[2:])  # preserve extra dimensions
    R[0:2, 0:2] = R_block
    R[2, 2] = 1

    # -----------------------------
    # P matrix: transformation from 3x3 tensor to 6x1 Voigt notation
    # -----------------------------
    P = np.zeros((6, 9))
    P[0, 0] = 1
    P[1, 4] = 1
    P[2, 8] = 1
    P[3, 1] = 1
    P[3, 3] = 1
    P[4, 2] = 1
    P[4, 6] = 1
    P[5, 5] = 1
    P[5, 7] = 1

    Q = np.zeros((9, 6))
    Q[0, 0] = 1
    Q[1, 3] = 0.5
    Q[2, 4] = 0.5
    Q[3, 3] = 0.5
    Q[4, 1] = 1
    Q[5, 5] = 0.5
    Q[6, 4] = 0.5
    Q[7, 5] = 0.5
    Q[8, 2] = 1

    # -----------------------------
    # Compute the transformation matrix T = P @ A @ A @ Q
    # Using einsum with indices to handle potential extra dimensions
    # -----------------------------
    T = np.einsum('ij,kj...,kl...,lm->im...', P, R, R, Q)

    # -----------------------------
    # Constitutive matrix in the local material frame (Fun case, Voigt 6x6)
    # -----------------------------
    E = material.E(xi3)
    nu = material.nu(xi3)

    factor = E / ((1 + nu) * (1 - 2 * nu))

    # shape (6,6,...) automaticamente
    C_local_material = np.zeros((6, 6) + E.shape)

    # Diagonal parte normal
    C_local_material[0, 0] = (1 - nu) * factor
    C_local_material[1, 1] = (1 - nu) * factor
    C_local_material[2, 2] = (1 - nu) * factor

    # Partes de acoplamento normal
    C_local_material[0, 1] = nu * factor
    C_local_material[0, 2] = nu * factor
    C_local_material[1, 0] = nu * factor
    C_local_material[1, 2] = nu * factor
    C_local_material[2, 0] = nu * factor
    C_local_material[2, 1] = nu * factor

    # Cisalhamentos
    C_local_material[3, 3] = (1 - 2 * nu) / 2 * factor
    C_local_material[4, 4] = (1 - 2 * nu) / 2 * factor
    C_local_material[5, 5] = (1 - 2 * nu) / 2 * factor

    # -----------------------------
    # Compute the constitutive tensor in the shell reciprocal frame
    # C_shell = T^T @ C_local @ T using einsum
    # -----------------------------
    C_shell_reciprocal_frame = np.einsum('ij...,jk...,kl...->il...', T, C_local_material, T)

    return C_shell_reciprocal_frame


@dispatch(MidSurfaceGeometry, LaminateOrthotropicMaterial, np.ndarray, np.ndarray, np.ndarray)
@cache_function
def constitutive_matrix_for_fosd2(mid_surface_geometry: MidSurfaceGeometry, material: LaminateOrthotropicMaterial, xi1,
                                 xi2,
                                 xi3):
    """
        Compute the constitutive tensor in the shell reciprocal frame for FOSD2.

        Parameters
        ----------
        mid_surface_geometry : MidSurfaceGeometry
            Object containing the shell geometry and metric properties.
        material : LaminateOrthotropicMaterial
        xi1, xi2, xi3 : arrays
            Coordinates in the curvilinear parametric space.

        Returns
        -------
        C_shell_reciprocal_frame : array
            Constitutive tensor in Voigt notation in the shell reciprocal frame.
            Shape: (6,6, ...) matching the shape of xi1, xi2, xi3
        """
    alpha = material.angle(xi3)
    T = transformation_matrix_fosd2_alpha(mid_surface_geometry, alpha, xi1, xi2, xi3)

    # -----------------------------
    # Constitutive matrix in the local material frame (Fun case, Voigt 6x6)
    # -----------------------------

    C_local_material = material.orthotropic_voigt_matrix(xi3)

    # -----------------------------
    # Compute the constitutive tensor in the shell reciprocal frame
    # C_shell = T^T @ C_local @ T using einsum
    # -----------------------------
    C_shell_reciprocal_frame = np.einsum('ij...,jk...,kl...->il...', T, C_local_material, T)

    return C_shell_reciprocal_frame