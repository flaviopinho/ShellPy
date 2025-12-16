import sympy as sym
import numpy as np

from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy.fosd_theory.fosd_kinetic_energy import fosd_kinetic_energy
from shellpy.fosd_theory.fosd_strain_tensor import fosd_linear_strain_components
from shellpy.fosd_theory2.fosd2_kinetic_energy import fosd2_kinetic_energy
from shellpy.fosd_theory2.fosd2_strain_vector import fosd2_linear_strain_vector
from shellpy import RectangularMidSurfaceDomain, Shell, ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial


def test_fosd_kinect_energy():
    a = 1
    b = 1
    h = 0.1

    U = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7])

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

    R_ = sym.Matrix([xi1_, xi2_, (xi1_ - a / 2) ** 2 + (xi2_ - b / 2) ** 2 - (xi1_ - a / 2) * (xi2_ - b / 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    material = IsotropicHomogeneousLinearElasticMaterial(10, 0.3, 1550)

    shell=Shell(mid_surface_geometry, ConstantThickness(h), edges, material, displacement_field, None)

    T = fosd2_kinetic_energy(shell, 40, 40, 10)

    T = np.einsum('ij, i, j->', T, U, U)
    print(T)

    T = fosd_kinetic_energy(shell, 40, 40, 10)

    T = np.einsum('ij, i, j->', T, U, U)
    print(T)


if __name__ == "__main__":
    test_fosd_kinect_energy()
