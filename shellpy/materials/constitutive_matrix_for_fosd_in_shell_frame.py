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

def constitutive_matrix_for_fosd_in_shell_frame(mid_surface_geometry, C_local, xi1, xi2, xi3):

    T = transformation_matrix_fosd2_global(mid_surface_geometry, xi1, xi2, xi3)

    C_shell_reciprocal_frame = np.einsum('ji...,jk...,kl...->il...', T, C_local, T)

    return C_shell_reciprocal_frame
