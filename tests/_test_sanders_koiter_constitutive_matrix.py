import numpy as np
from shellpy import MidSurfaceGeometry, xi1_, xi2_
import sympy as sym

from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.sanders_koiter.plane_stress_constitutive_matrix_in_material_frame import \
    constitutive_matrix_in_material_frame
from shellpy.sanders_koiter.plane_stress_constitutive_matrix_in_shell_frame import \
    plane_stress_constitutive_matrix_in_shell_frame


def _test_constitutive_matrix_for_sanders_koiter():
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.001

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2) - xi1_])
    mid_surface = MidSurfaceGeometry(R_)

    material = IsotropicHomogeneousLinearElasticMaterial(10, 0.3, 0)

    xi1 = 0.02
    xi2 = 0
    xi3 = 0.01

    C_local = constitutive_matrix_in_material_frame(mid_surface, material, (xi1, xi2, xi3))
    C = plane_stress_constitutive_matrix_in_shell_frame(mid_surface, C_local, (xi1, xi2, xi3))

    print("\n Constitutive matrix")
    print(C)

    C_ext_expected = np.array([[4.58787362600000, 1.48741455400000, 0.944965324800000],
                               [1.48741455300000, 4.42146504000000, 0.927669433400000],
                               [0.944965324800000, 0.927669433500000, 1.70287786000000]])

    assert np.allclose(
        C, C_ext_expected, rtol=1e-7, atol=1e-7
    ), f"Expected {C_ext_expected}, got {C}"


def _test_constitutive_matrix_for_sanders_koiter_grid():
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.001

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2) - xi1_])
    mid_surface = MidSurfaceGeometry(R_)

    material = IsotropicHomogeneousLinearElasticMaterial(10, 0.3, 0)

    xi1_lin = np.linspace(0, a, 4)
    xi2_lin = np.linspace(0, b, 6)

    xi1, xi2 = np.meshgrid(xi1_lin, xi2_lin, indexing='ij')

    xi3 = np.linspace(-h / 2, h / 2, 2)

    C_local = constitutive_matrix_in_material_frame(mid_surface, material, (xi1, xi2, xi3))
    C_grid = plane_stress_constitutive_matrix_in_shell_frame(mid_surface, C_local, (xi1, xi2, xi3))

    for i in range(xi1.shape[0]):
        for j in range(xi2.shape[1]):
            for k in range(xi3.shape[-1]):
                x = xi1[i, j]
                y = xi2[i, j]
                z = xi3[k]
                C = plane_stress_constitutive_matrix_in_shell_frame(mid_surface, C_local, (x, y, z))

                assert np.allclose(C_grid[:, :, i, j, k], C, rtol=1e-8,
                                   atol=1e-12), f"Expected {C}, got {C_grid[:, :, i, j, k]}"


if __name__ == "__main__":
    _test_constitutive_matrix_for_sanders_koiter()
    _test_constitutive_matrix_for_sanders_koiter_grid()
