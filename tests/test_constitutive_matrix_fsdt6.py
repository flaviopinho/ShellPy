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
                               [0., 0., 0., 2.03305533750000, 0.426556192583333, 0.],
                               [0., 0., 0., 0.426556192583333, 2.07096054750000, 0.],
                               [1.15758252300000, 1.13639505600000, 0.767801146800000, 0., 0., 1.74667067000000]])

    C_app_expected = np.array([[5.620067334869208, 2.500741843871445, 3.72770316644098, 0.5289715655 * 10 ** (-9),
                                -0.1580342299 * 10 ** (-8), 1.1575287386189732],
                               [2.500741843871445, 5.416188163762124, 3.6594636280578188, 0.2105115636 * 10 ** (-8),
                                -0.2168182350 * 10 ** (-9), 1.1363389541078996],
                               [3.7277031664409797, 3.659463628057819, 13.461538473162124, 0.2723166258 * 10 ** (-8),
                                -0.1854343629 * 10 ** (-8), 0.7677707912735984],
                               [0.5289716969 * 10 ** (-9), 0.2105115498 * 10 ** (-8), 0.2723166454 * 10 ** (-8),
                                2.033035348472699, 0.42653932841642994, 0.2178733267 * 10 ** (-10)],
                               [-0.1580341857 * 10 ** (-8), -0.2168185464 * 10 ** (-9), -0.1854343405 * 10 ** (-8),
                                0.42653932841642994, 2.070946203117372, 0.2231640816 * 10 ** (-9)],
                               [1.1575287386189734, 1.1363389541079003, 0.7677707912735986, 0.2178730456 * 10 ** (-10),
                                0.2231640265 * 10 ** (-9), 1.7466307857569117]])

    assert np.allclose(
        C, C_app_expected, rtol=1e-8, atol=1e-8
    ), f"Expected {C_app_expected}, got {C}"

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
