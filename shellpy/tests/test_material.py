import numpy as np
from shellpy import MidSurfaceGeometry, xi1_, xi2_
import sympy as sym

from shellpy.materials.constitutive_tensor_koiter import plane_stress_constitutive_tensor_for_koiter_theory
from shellpy.materials.linear_elastic_material import LinearElasticMaterial

if __name__ == "__main__":
    R = 1
    a = 0.1
    b = 0.1

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface = MidSurfaceGeometry(R_)

    material = LinearElasticMaterial(10, 0.2, 0)

    C = plane_stress_constitutive_tensor_for_koiter_theory(mid_surface, material, 0, 0, 0)

    idn = np.zeros((2,) * 4)
    idn[tuple([np.arange(2)] * 4)] = 1

    print(idn)