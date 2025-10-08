import time
import sympy as sym
import numpy as np

from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy import RectangularMidSurfaceDomain, xi1_, xi2_, MidSurfaceGeometry


def setup_mid_surface():
    """
    Sets up a MidSurfaceGeometry object and meshgrid for testing.

    Returns
    -------
    mid_surface_geometry : MidSurfaceGeometry
        The midsurface geometry object.
    x, y : ndarray
        Meshgrid of xi1 and xi2 coordinates for testing.
    """
    R = 0.1
    a = 0.1
    b = 0.1

    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    xi1 = np.linspace(*edges.edges["xi1"], 500)
    xi2 = np.linspace(*edges.edges["xi2"], 500)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    return mid_surface_geometry, x, y


def test_natural_base_performance():
    """
    Tests the performance and caching of the natural_base function.
    Prints timing for first and subsequent calls.
    """
    mid_surface_geometry, x, y = setup_mid_surface()

    for i in range(1, 4):
        start = time.time()
        data = mid_surface_geometry.natural_base(x, y)
        end = time.time()
        print(f'({i}) natural_base computed in {end - start:.2f} s, output shape {np.shape(data)}')


def test_christoffel_symbols_first_derivative_performance():
    """
    Tests the performance and caching of the christoffel_symbols_first_derivative function.
    Prints timing for first and subsequent calls.
    """
    mid_surface_geometry, x, y = setup_mid_surface()

    for i in range(1, 4):
        start = time.time()
        data = mid_surface_geometry.christoffel_symbols_first_derivative(x, y)
        end = time.time()
        print(
            f'({i}) christoffel_symbols_first_derivative computed in {end - start:.2f} s, output shape {np.shape(data)}')


if __name__ == "__main__":
    print("=== Testing natural_base performance ===")
    test_natural_base_performance()

    print("\n=== Testing christoffel_symbols_first_derivative performance ===")
    test_christoffel_symbols_first_derivative_performance()
