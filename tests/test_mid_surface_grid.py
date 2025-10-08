import numpy as np
import sympy as sym

from shellpy import MidSurfaceGeometry, xi1_, xi2_

"""
Grid-based test scripts for the MidSurfaceGeometry class from shellpy.

These functions verify the consistency and correctness of the midsurface computations 
across discrete grids of xi1, xi2, and xi3 coordinates:

1. test_mid_surface_grid1():
   - Tests the curvature tensor in mixed components (K^alpha_beta) on a 2D grid of (xi1, xi2).
   - Compares the tensor computed for each grid point individually against the vectorized computation.

2. test_mid_surface_grid2():
   - Tests the shifter tensor (Upsilon^alpha_beta) on a 3D grid of (xi1, xi2, xi3).
   - Ensures that the tensor computed pointwise matches the precomputed grid values.

3. test_mid_surface_grid3():
   - Similar to grid2 but with variable thickness (xi3-dependent height) across the midsurface.
   - Confirms that the shifter tensor handles non-uniform grids correctly.

All tests use strict numerical tolerance (np.allclose) to validate the correctness of the 
vectorized and scalar implementations, ensuring accurate geometry computations 
for shell analyses.
"""


def test_mid_surface_grid1():
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.01

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2) - xi1_])
    midsurface = MidSurfaceGeometry(R_)

    xi1_lin = np.linspace(0, a, 5)
    xi2_lin = np.linspace(0, b, 7)

    xi1, xi2 = np.meshgrid(xi1_lin, xi2_lin, indexing='ij')

    K_mixed_grid = midsurface.curvature_tensor_mixed_components(xi1, xi2)

    for i in range(xi1.shape[0]):
        for j in range(xi2.shape[1]):
            x = xi1[i, j]
            y = xi2[i, j]
            K_mixed = midsurface.curvature_tensor_mixed_components(x, y)

            assert np.allclose(K_mixed_grid[:, :, i, j], K_mixed, rtol=1e-8,
                               atol=1e-12), f"Expected {K_mixed}, got {K_mixed_grid[:, :, i, j]}"


def test_mid_surface_grid2():
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.01

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2) - xi1_])
    midsurface = MidSurfaceGeometry(R_)

    xi1_lin = np.linspace(0, a, 4)
    xi2_lin = np.linspace(0, b, 6)

    xi1, xi2 = np.meshgrid(xi1_lin, xi2_lin, indexing='ij')

    h = 0.001
    xi3 = np.linspace(-h / 2, h / 2, 2)

    shifter_tensor_grid = midsurface.shifter_tensor(xi1, xi2, xi3)

    for i in range(xi1.shape[0]):
        for j in range(xi2.shape[1]):
            for k in range(xi3.shape[-1]):
                x = xi1[i, j]
                y = xi2[i, j]
                z = xi3[k]
                shifter_tensor = midsurface.shifter_tensor(x, y, z)

                assert np.allclose(shifter_tensor_grid[:, :, i, j, k], shifter_tensor, rtol=1e-8,
                                   atol=1e-12), f"Expected {shifter_tensor}, got {shifter_tensor_grid[:, :, i, j, k]}"


def test_mid_surface_grid3():
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

    shifter_tensor_grid = midsurface.shifter_tensor(xi1, xi2, xi3)

    for i in range(xi1.shape[0]):
        for j in range(xi2.shape[1]):
            for k in range(xi3.shape[-1]):
                x = xi1[i, j]
                y = xi2[i, j]
                z = xi3[i, j, k]
                shifter_tensor = midsurface.shifter_tensor(x, y, z)

                assert np.allclose(shifter_tensor_grid[:, :, i, j, k], shifter_tensor, rtol=1e-8,
                                   atol=1e-12), f"Expected {shifter_tensor}, got {shifter_tensor_grid[:, :, i, j, k]}"


if __name__ == "__main__":
    test_mid_surface_grid1()
    test_mid_surface_grid2()
    test_mid_surface_grid3()
