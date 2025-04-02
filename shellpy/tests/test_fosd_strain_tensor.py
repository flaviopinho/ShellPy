import sympy as sym
import numpy as np

from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy.fosd_theory.fosd_strain_tensor import fosd_linear_strain_components, fosd_nonlinear_strain_components
from shellpy.koiter_shell_theory.koiter_strain_tensor import koiter_linear_strain_components, \
    koiter_nonlinear_strain_components_total
from shellpy import RectangularMidSurfaceDomain
from shellpy import MidSurfaceGeometry, xi1_, xi2_

if __name__ == "__main__":
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.0001

    E = 10
    nu = 0.2

    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    expansion_size = {"u1": (1, 1),
                      "u2": (1, 1),
                      "u3": (1, 1),
                      "v1": (1, 1),
                      "v2": (1, 1),
                      "v3": (1, 1)}

    boundary_conditions_u1 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_u2 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_u3 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_v1 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_v2 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_v3 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3,
                           "v1": boundary_conditions_v1,
                           "v2": boundary_conditions_v2,
                           "v3": boundary_conditions_v3}

    displacement_field = EigenFunctionExpansion(expansion_size, edges, boundary_conditions)

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    epsilon0, epsilon1, epsilon2 = fosd_linear_strain_components(mid_surface_geometry, displacement_field, 1, a / 4, b / 3)

    print("epsilon_lin = \n", epsilon0)

    epsilon0, epsilon1, epsilon2  = fosd_nonlinear_strain_components(mid_surface_geometry, displacement_field, 2, 1, a / 4, b / 3)

    print("epsilon_nonlin = \n", epsilon0)

    xi1 = np.linspace(0, a, 3)
    xi2 = np.linspace(0, b, 3)

    x, y = np.meshgrid(xi1, xi2)

    epsilon0, epsilon1, epsilon2 = fosd_linear_strain_components(mid_surface_geometry, displacement_field, 1, x, y)

    print("epsilon_lin = \n", epsilon0)
