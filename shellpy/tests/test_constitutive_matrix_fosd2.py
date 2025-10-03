import numpy as np
from shellpy import MidSurfaceGeometry, xi1_, xi2_
import sympy as sym

from shellpy.materials.constitutive_matrix_fosd2 import constitutive_matrix_for_fosd2
from shellpy.materials.linear_elastic_material import LinearElasticMaterial

if __name__ == "__main__":
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.001

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface = MidSurfaceGeometry(R_)

    material = LinearElasticMaterial(10, 0.3, 0)

    xi1 = np.atleast_2d(0)
    xi2 = np.atleast_2d(0)
    xi3 = np.atleast_1d(h)

    C = constitutive_matrix_for_fosd2(mid_surface, material, xi1, xi2, xi3)

    print(np.squeeze(C))

    xi1 = np.linspace(0, a, 10)
    xi2 = np.linspace(0, b, 10)

    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    xi3 = np.linspace(-h/2, h/2, 5)

    C = constitutive_matrix_for_fosd2(mid_surface, material, x, y, xi3)
    print(np.shape(C))
