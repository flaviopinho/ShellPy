import numpy as np
from shellpy import MidSurfaceGeometry, xi1_, xi2_
import sympy as sym

from shellpy.materials.constitutive_tensor_koiter import plane_stress_constitutive_tensor_for_koiter_theory
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial

if __name__ == "__main__":
    R = 1
    a = 0.1
    b = 0.1

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface = MidSurfaceGeometry(R_)

    material = IsotropicHomogeneousLinearElasticMaterial(10, 0.2, 0)

    C = plane_stress_constitutive_tensor_for_koiter_theory(mid_surface, material, 0, 0, 0)
    print(C)
