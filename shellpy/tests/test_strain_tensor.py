import sympy as sym
import numpy as np

from expansions.eigen_function_expansion import EigenFunctionExpansion
from koiter_shell_theory.koiter_strain_tensor import koiter_linear_strain_components, \
    koiter_nonlinear_strain_components_total
from mid_surface_domain import RectangularMidSurfaceDomain
from midsurface_geometry import MidSurfaceGeometry, xi1_, xi2_

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
                      "u3": (1, 1)}

    boundary_conditions_u1 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_u2 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_u3 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3}

    displacement_field = EigenFunctionExpansion(expansion_size, edges, boundary_conditions)

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    gamma, rho = koiter_linear_strain_components(mid_surface_geometry, displacement_field, 1, a / 4, b / 3)

    gamma_nl = koiter_nonlinear_strain_components_total(mid_surface_geometry, displacement_field, 1, 0, a / 4, b / 3)
    print("gamma_lin = \n", gamma)
    print("rho_lin = \n", rho)
    print("gamma_nonlin = \n", gamma_nl)

    xi1 = np.linspace(*edges.edges["xi1"],100)
    xi2 = np.linspace(*edges.edges["xi2"],100)
    x, y = np.meshgrid(xi1, xi2, indexing='xy')

    gamma, rho = koiter_linear_strain_components(mid_surface_geometry, displacement_field, 1, x, y)

    gamma_nl = koiter_nonlinear_strain_components_total(mid_surface_geometry, displacement_field, 1, 0, x, y)

    print("\nx, y = ", x[1, 2], ",", y[1, 2])
    print("gamma = \n", gamma[:, :, 1, 2])
    print("rho = \n", rho[:, :, 1, 2])
    print("gamma_nl = \n", gamma_nl[:, :, 1, 2])

