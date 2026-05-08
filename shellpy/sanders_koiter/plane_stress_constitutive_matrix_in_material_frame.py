import numpy as np
from multipledispatch import dispatch

from .plane_stress_transformation_matrix import plane_stress_transformation_matrix_rotation
from ..midsurface_geometry import MidSurfaceGeometry
from ..cache_decorator import cache_function
from ..materials.functionally_graded_material import FunctionallyGradedMaterial
from ..materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from ..materials.laminate_orthotropic_material import LaminateOrthotropicMaterial


@dispatch(MidSurfaceGeometry, IsotropicHomogeneousLinearElasticMaterial, object)
@cache_function
def constitutive_matrix_in_material_frame(mid_surface_geometry, material, position=None):
    """
    Computes the 3x3 Constitutive Matrix (C) for Isotropic Homogeneous materials.

    Assumes Plane Stress condition and Classic Voigt order: [11, 22, 12].
    """
    E = material.E
    nu = material.nu

    # Plane stress stiffness factor
    Q = E / (1 - nu ** 2)

    C = np.zeros((3, 3))

    # Normal components
    C[0, 0] = Q
    C[1, 1] = Q

    # Coupling components (Poisson effect)
    C[0, 1] = nu * Q
    C[1, 0] = nu * Q

    # Shear component G = E / (2*(1+nu))
    # Equivalent to Q * (1 - nu) / 2.0
    C[2, 2] = Q * (1 - nu) / 2.0

    return C


@dispatch(MidSurfaceGeometry, FunctionallyGradedMaterial, object)
@cache_function
def constitutive_matrix_in_material_frame(mid_surface_geometry: MidSurfaceGeometry,
                                          material: FunctionallyGradedMaterial, position):
    """
    Computes the 3x3 Constitutive Matrix (C) for Functionally Graded Materials (FGM).

    The properties E and nu vary through the thickness coordinate (xi3).
    The returned matrix maintains the shape of the input coordinates for
    vectorized integration.
    """
    xi3 = position[2]

    # Evaluate properties at the specific thickness coordinate
    E = material.E(xi3)
    nu = material.nu(xi3)

    Q = E / (1 - nu ** 2)

    # Initialize a constitutive tensor that matches the shape of the integration points
    C = np.zeros((3, 3) + np.shape(E))

    C[0, 0] = Q
    C[1, 1] = Q

    C[0, 1] = nu * Q
    C[1, 0] = nu * Q

    C[2, 2] = Q * (1 - nu) / 2.0

    return C


@dispatch(MidSurfaceGeometry, LaminateOrthotropicMaterial, object)
@cache_function
def constitutive_matrix_in_material_frame(mid_surface_geometry: MidSurfaceGeometry,
                                          material: LaminateOrthotropicMaterial, position):
    """
    Computes the 3x3 Constitutive Matrix (C) for Laminated Orthotropic materials.

    This follows a two-step process:
    1. Define the reduced stiffness matrix (Q) in the local fiber frame.
    2. Rotate the matrix to the shell's local orthogonal frame (Q_bar)
       using the fiber orientation angle (alpha).
    """
    xi1 = position[0]
    xi2 = position[1]
    xi3 = position[2]

    # Identify which lamina (layer) exists at each integration point
    index = material.lamina_index(xi1, xi2, xi3)
    shape_out = (3, 3) + np.shape(index)
    C_all = np.zeros(shape_out, dtype=float)

    C_laminas = []
    for lamina in material.laminas:
        E1, E2 = lamina.E_11, lamina.E_22
        nu12 = lamina.nu_12
        G12 = lamina.G_12

        # Reciprocal Poisson's ratio
        nu21 = nu12 * E2 / E1

        # Reduced stiffness matrix (Q) in the local FIBER system (orthotropic)
        Q = np.zeros((3, 3))
        Q[0, 0] = E1 / (1 - nu12 * nu21)
        Q[1, 1] = E2 / (1 - nu12 * nu21)
        Q[0, 1] = nu12 * E2 / (1 - nu12 * nu21)
        Q[1, 0] = Q[0, 1]
        Q[2, 2] = G12

        # 1. Obtain the 3x3 rotation matrix T based on the fiber angle
        T = plane_stress_transformation_matrix_rotation(lamina.angle)

        # 2. Perform a congruent transformation to the shell's local orthogonal system (Q_bar)
        # Formula: Q_bar = T.T * Q * T
        C_rot = T.T @ Q @ T

        C_laminas.append(C_rot)

    C_laminas = np.array(C_laminas)

    # Fill the global C_all matrix using masks to assign layer properties
    # to the correct vertical integration points
    for i_lam, C in enumerate(C_laminas):
        mask = (index == i_lam)
        if np.any(mask):
            C_all[:, :, mask] = C[:, :, np.newaxis]

    return C_all