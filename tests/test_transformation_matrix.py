import numpy as np
from shellpy import MidSurfaceGeometry, xi1_, xi2_
import sympy as sym
from shellpy.materials.transformation_matrix_fosd2 import transformation_matrix_fosd2_local, \
    transformation_matrix_fosd2_global

"""
================================================================================
TEST MODULE: Transformation Matrices for FOSD2 Theory
================================================================================
This test script validates the implementation of the local and global 
transformation matrices used in the First-Order Shear Deformation Theory (FOSD2) 
for curved shell geometries.

Functions tested:
    - transformation_matrix_fosd2_local
    - transformation_matrix_fosd2_global

The tests verify:
    1. Correct shape and numerical values of transformation matrices at given
       coordinates.
    2. Consistency between matrix values computed individually and over 
       structured grids of points (xi1, xi2, xi3, alpha).
    3. Numerical accuracy using np.allclose() with defined tolerances.

Dependencies:
    - numpy
    - sympy
    - shellpy.geometry.MidSurfaceGeometry
    - shellpy.materials.transformation_matrix_fosd2

Run as standalone:
    python test_transformation_matrix_fosd2.py
================================================================================
"""


def test_transformation_matrix_local():
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.001

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2) - xi1_])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    T = transformation_matrix_fosd2_local(mid_surface_geometry, a / 4, b / 3, -h / 2, np.pi / 3)

    print("Shape of transformation matrix (local):", T.shape)
    print("Transformation matrix (local) at xi3=-h/2:")
    print(T)

    T_expected = np.array([[0.226319142413059, 0.742187042187636, 1.67765607400801 * 10 ** (-20),
                            -1.11585599401362 * 10 ** (-10), -6.16186403561092 * 10 ** (-11), 0.409842817306819],
                           [0.431921759045108, 0.246881077931896, 8.74287189847664 * 10 ** (-22),
                            -1.46916630730370 * 10 ** (-11), 1.94325412890236 * 10 ** (-11), -0.326547560786016],
                           [4.55304948350558 * 10 ** (-20), 1.59101235680554 * 10 ** (-20), 1.,
                            -1.26135338300000 * 10 ** (-10), 2.13378759100000 * 10 ** (-10),
                            -2.69146019651127 * 10 ** (-20)],
                           [-2.80468261443975 * 10 ** (-10), -1.25346056284371 * 10 ** (-10),
                            -5.91366955400000 * 10 ** (-11), 0.496871289100000, -0.657207546400000,
                            1.88918875299052 * 10 ** (-10)],
                           [2.03021403253076 * 10 ** (-10), -2.17331889531333 * 10 ** (-10),
                            -2.59048727000000 * 10 ** (-10), 0.861502781300000, 0.475730115100000,
                            1.23820015427356 * 10 ** (-10)],
                           [-0.625306843386921, 0.856111995015533, 7.65964284931179 * 10 ** (-21),
                            -8.98301513044026 * 10 ** (-11), 7.10578356469178 * 10 ** (-11), -0.329809493561520]])

    assert np.allclose(T, T_expected, rtol=1e-7,
                       atol=1e-7), f"Expected {T_expected}, got {T}"


def test_transformation_matrix_global():
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.001

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2) - xi1_])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    T = transformation_matrix_fosd2_global(mid_surface_geometry, a / 4, b / 3, -h / 2)

    print("Shape of transformation matrix (global):", T.shape)
    print("Transformation matrix (global) at xi3=-h/2:")
    print(T)

    T_expected = np.array([[0.430687942718529, 0.00681917577622781, 0.345698377526722, 0.0485528372179369,
                            0.385860237674365, 0.0541935123986219],
                           [0.00681917576631841, 0.969893140512478, 0.0193867147677984, -0.137124183389946,
                            -0.0114978874378339, 0.0813257142587826],
                           [0.220733782503601, 0.0123558027066543, 0.634914907733362, 0.0885713460181510,
                            -0.374362350072640, -0.0522239702369327],
                           [-0.0775943931081713, 0.218941163701205, -0.221891092367853, 0.769251977737807,
                            0.131216049162062, -0.453517512956347],
                           [-0.616659966837333, 0.0183582559643412, 0.936993177074309, 0.131155471095558,
                            0.246686876706306, 0.0341513828991992],
                           [0.108387024718492, 0.162651428635745, -0.163730948092305, 0.567544844180296,
                            -0.0428234414645588, 0.653132784899328]])

    assert np.allclose(T, T_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {T_expected}, got {T}"


def test_transformation_matrix_grid1():
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.01

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2) - xi1_])
    midsurface = MidSurfaceGeometry(R_)

    xi1_lin = np.linspace(0, a, 4)
    xi2_lin = np.linspace(0, b, 6)

    xi1, xi2 = np.meshgrid(xi1_lin, xi2_lin, indexing='ij')

    h = 0.001 + xi1 / 100 + xi2 / 100
    xi3 = np.linspace(-h / 2, h / 2, 2).transpose((1, 2, 0))
    alpha = np.random.rand(*xi3.shape)

    T_grid = transformation_matrix_fosd2_local(midsurface, xi1, xi2, xi3, alpha)

    for i in range(xi1.shape[0]):
        for j in range(xi2.shape[1]):
            for k in range(xi3.shape[-1]):
                x = xi1[i, j]
                y = xi2[i, j]
                z = xi3[i, j, k]
                a = alpha[i, j, k]
                T = transformation_matrix_fosd2_local(midsurface, x, y, z, a)

                assert np.allclose(T_grid[:, :, i, j, k], T, rtol=1e-8,
                                   atol=1e-12), f"Expected {T}, got {T_grid[:, :, i, j, k]}"


def test_transformation_matrix_grid2():
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.01

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2) - xi1_])
    midsurface = MidSurfaceGeometry(R_)

    xi1_lin = np.linspace(0, a, 4)
    xi2_lin = np.linspace(0, b, 6)

    xi1, xi2 = np.meshgrid(xi1_lin, xi2_lin, indexing='ij')

    h = 0.001 + xi1 / 100 + xi2 / 100
    xi3 = np.linspace(-h / 2, h / 2, 2).transpose((1, 2, 0))

    T_grid = transformation_matrix_fosd2_global(midsurface, xi1, xi2, xi3)

    for i in range(xi1.shape[0]):
        for j in range(xi2.shape[1]):
            for k in range(xi3.shape[-1]):
                x = xi1[i, j]
                y = xi2[i, j]
                z = xi3[i, j, k]
                T = transformation_matrix_fosd2_global(midsurface, x, y, z)

                assert np.allclose(T_grid[:, :, i, j, k], T, rtol=1e-8,
                                   atol=1e-12), f"Expected {T}, got {T_grid[:, :, i, j, k]}"


if __name__ == "__main__":
    test_transformation_matrix_local()
    test_transformation_matrix_global()
    test_transformation_matrix_grid1()
    test_transformation_matrix_grid2()
