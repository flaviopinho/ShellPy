import numpy as np
from shellpy import MidSurfaceGeometry, xi1_, xi2_
import sympy as sym
from shellpy.materials.laminate_orthotropic_material import Lamina, LaminateOrthotropicMaterial
from shellpy.materials.transformation_matrix_fosd2 import transformation_matrix_fosd2_alpha, \
    transformation_matrix_fosd2_global

if __name__ == "__main__":

    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.001

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    T = transformation_matrix_fosd2_alpha(mid_surface_geometry, np.pi/3, a/4, b/3, -h/2)

    print("Shape of transformation matrix (local):", T.shape)
    print("Transformation matrix (local) at xi3=-h/2:")
    print(np.squeeze(T))

    T = transformation_matrix_fosd2_global(mid_surface_geometry, a / 4, b / 3, -h / 2)

    print("Shape of transformation matrix (global):", T.shape)
    print("Transformation matrix (global) at xi3=-h/2:")
    print(np.squeeze(T))
