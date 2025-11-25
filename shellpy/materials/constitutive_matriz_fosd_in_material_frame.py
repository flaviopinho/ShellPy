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
def constitutive_matrix_for_fosd_in_material_frame(mid_surface_geometry, material, xi1, xi2, xi3):

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

    C = C_local_material[:, :, None, None, None]
    C = np.repeat(C, np.shape(xi1)[0],axis=2)
    C = np.repeat(C, np.shape(xi1)[1], axis=3)
    C = np.repeat(C, np.shape(xi3)[-1], axis=4)

    return C


@dispatch(MidSurfaceGeometry, FunctionallyGradedMaterial, object, object, object)
@cache_function
def constitutive_matrix_for_fosd_in_material_frame(mid_surface_geometry: MidSurfaceGeometry, material: FunctionallyGradedMaterial, xi1,
                                  xi2,
                                  xi3):
    # -----------------------------
    # Constitutive matrix in the local material frame (isotropic case, Voigt 6x6)
    # -----------------------------
    E = material.E(xi3)
    nu = material.nu(xi3)

    # Common factor
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

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
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu

    return C