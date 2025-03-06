import numpy as np
from shellpy import LinearElasticMaterial
from shellpy import MidSurfaceGeometry, xi1_, xi2_
import sympy as sym

if __name__ == "__main__":
    R = 1
    a = 0.1
    b = 0.1

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface = MidSurfaceGeometry(R_)

    material = LinearElasticMaterial(10, 0.2, 0)

    C = material.plane_stress_constitutive_tensor_for_koiter_theory(mid_surface.metric_tensor_contravariant_components(0, 0))

    idn = np.zeros((2,) * 4)
    idn[tuple([np.arange(2)] * 4)] = 1

    print(idn)