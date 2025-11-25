import numpy as np
from multipledispatch import dispatch

from shellpy import MidSurfaceGeometry, cache_function
from shellpy.fosd_theory2.shear_correction_factor import shear_correction_factor
from shellpy.materials.functionally_graded_material import FunctionallyGradedMaterial
from shellpy.materials.laminate_orthotropic_material import LaminateOrthotropicMaterial
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.materials.orthotropic_material import OrthotropicMaterial
from shellpy.materials.transformation_matrix_fosd2 import transformation_matrix_fosd2_global, \
    transformation_matrix_fosd2_local


@dispatch(MidSurfaceGeometry, IsotropicHomogeneousLinearElasticMaterial, object, object, object)
@cache_function
def constitutive_matrix_for_fosd2(mid_surface_geometry, material, xi1, xi2, xi3):
    """
    Compute the constitutive tensor in the shell reciprocal frame for FOSD2.

    Parameters
    ----------
    mid_surface_geometry : MidSurfaceGeometry
        Object containing the shell geometry and metric properties.
    material : IsotropicHomogeneousLinearElasticMaterial
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


@dispatch(MidSurfaceGeometry, FunctionallyGradedMaterial, object, object, object)
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

    T = transformation_matrix_fosd2_global(mid_surface_geometry, xi1, xi2, xi3)

    # -----------------------------
    # Constitutive matrix in the local material frame (isotropic case, Voigt 6x6)
    # -----------------------------
    E = material.E(xi3)
    nu = material.nu(xi3)

    # Common factor
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2*(1+nu))

    # Initialize matrix with zeros, shape (6,6)+E.shape
    C = np.zeros((6, 6) + E.shape)

    # Fill in normal components
    C[0, 0] = lambda_ + 2 * mu
    C[1, 1] = lambda_ + 2 * mu
    C[2, 2] = lambda_ + 2 * mu

    C[0, 1] = lambda_
    C[1, 0] = lambda_
    C[0, 2] = lambda_
    C[2, 0] = lambda_
    C[1, 2] = lambda_
    C[2, 1] = lambda_

    # Shear components
    C[3, 3] = mu * 0.145
    C[4, 4] = mu * 0.145
    C[5, 5] = mu



    # -----------------------------
    # Compute the constitutive tensor in the shell reciprocal frame
    # C_shell = T^T @ C_local @ T using einsum
    # -----------------------------
    C_shell_reciprocal_frame = np.einsum('ji...,jk...,kl...->il...', T, C, T)

    return C_shell_reciprocal_frame


@dispatch(MidSurfaceGeometry, LaminateOrthotropicMaterial, object, object, object)
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
    alpha = material.angle(xi1, xi2, xi3)
    T = transformation_matrix_fosd2_local(mid_surface_geometry, xi1, xi2, xi3, alpha)

    # -----------------------------
    # Constitutive matrix in the local material frame (Fun case, Voigt 6x6)
    # -----------------------------

    C_local_material = material.orthotropic_voigt_matrix(xi1, xi2, xi3)

    # -----------------------------
    # Compute the constitutive tensor in the shell reciprocal frame
    # C_shell = T^T @ C_local @ T using einsum
    # -----------------------------
    C_shell_reciprocal_frame = np.einsum('ji...,jk...,kl...->il...', T, C_local_material, T)

    return np.squeeze(C_shell_reciprocal_frame)



@dispatch(MidSurfaceGeometry, OrthotropicMaterial, object, object, object)
@cache_function
def constitutive_matrix_for_fosd2(mid_surface_geometry: MidSurfaceGeometry, material: OrthotropicMaterial, xi1,
                                 xi2,
                                 xi3):
    """
    Compute the constitutive tensor in the shell reciprocal frame for FOSD2.

    Parameters
    ----------
    mid_surface_geometry : MidSurfaceGeometry
        Object containing the shell geometry and metric properties.
    material : OrthotropicMaterial.
    xi1, xi2, xi3 : arrays
        Coordinates in the curvilinear parametric space.

    Returns
    -------
    C_shell_reciprocal_frame : array
        Constitutive tensor in Voigt notation in the shell reciprocal frame.
        Shape: (6,6, ...) matching the shape of xi1, xi2, xi3
    """

    T = transformation_matrix_fosd2_global(mid_surface_geometry, xi1, xi2, xi3, material.material_base)

    # -----------------------------
    # Constitutive matrix in the local material frame (isotropic case, Voigt 6x6)
    # -----------------------------
    C_local_material = material.orthotropic_voigt_matrix()

    # -----------------------------
    # Compute the constitutive tensor in the shell reciprocal frame
    # C_shell = T^T @ C_local @ T using einsum
    # -----------------------------
    C_shell_reciprocal_frame = np.einsum('ji...,jk,kl...->il...', T, C_local_material, T)

    return C_shell_reciprocal_frame