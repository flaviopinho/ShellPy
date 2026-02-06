import numpy as np
from shellpy import MidSurfaceGeometry, xi1_, xi2_
import sympy as sym

from shellpy.fsdt6.constitutive_matrix_in_shell_frame import constitutive_matrix_in_shell_frame
from shellpy.fsdt6.constitutive_matriz_in_material_frame import constitutive_matrix_in_material_frame
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial


def test_constitutive_matrix_for_fsdt6():
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
    C = constitutive_matrix_in_shell_frame(mid_surface, C_local, (xi1, xi2, xi3))

    print("\n Constitutive matrix")
    print(C)

    C_ext_expected = np.array([[5.62014519200000, 2.50079224500000, 3.72772898600000, 0., 0., 1.15758252300000],
                               [2.50079224500000, 5.41629467500000, 3.65949960900000, 0., 0., 1.13639505600000],
                               [3.72772898600000, 3.65949960900000, 13.4615384600000, 0., 0., 0.767801146800000],
                               [0., 0., 0., 2.43966640500000, 0.511867431100000, 0.],
                               [0., 0., 0., 0.511867431100000, 2.48515265700000, 0.],
                               [1.15758252300000, 1.13639505600000, 0.767801146800000, 0., 0., 1.74667067000000]])

    assert np.allclose(
        C, C_ext_expected, rtol=1e-7, atol=1e-7
    ), f"Expected {C_ext_expected}, got {C}"


def test_constitutive_matrix_for_fsdt6_grid():
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
    C_grid = constitutive_matrix_in_shell_frame(mid_surface, C_local, (xi1, xi2, xi3))

    for i in range(xi1.shape[0]):
        for j in range(xi2.shape[1]):
            for k in range(xi3.shape[-1]):
                x = xi1[i, j]
                y = xi2[i, j]
                z = xi3[k]
                C = constitutive_matrix_in_shell_frame(mid_surface, C_local, (x, y, z))

                assert np.allclose(C_grid[:, :, i, j, k], C, rtol=1e-8,
                                   atol=1e-12), f"Expected {C}, got {C_grid[:, :, i, j, k]}"


if __name__ == "__main__":
    test_constitutive_matrix_for_fsdt6()
    test_constitutive_matrix_for_fsdt6_grid()
