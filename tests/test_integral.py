import numpy as np

from shellpy import RectangularMidSurfaceDomain
from shellpy.numeric_integration.numeric_integration import simple_integral, double_integral


def func(xi1, xi2):
    return xi1 ** 2 + xi1 * xi2 + xi2 ** 2


def func2(xi1, xi2):
    return np.sin(11 * xi1 * np.pi / 5) * np.sin(9 * xi2 * np.pi / 3) * xi1 ** 3


def func3(xi3):
    return np.sin(11 * xi3 * np.pi / 5) * xi3 ** 3


def test_integrals():
    domain = RectangularMidSurfaceDomain(-2, 2, -2, 2)
    integral_result = double_integral(func, domain, 100, 100)
    print("integral result = ", integral_result)
    integral_result_excepted = 42.66666667

    assert np.allclose(
        integral_result, integral_result_excepted, rtol=1e-8, atol=1e-12
    ), f"Expected {integral_result_excepted}, got {integral_result}"

    integral_result = double_integral(func, ((-2, 2), (-2, 2)), 100, 100)
    print("integral result = ", integral_result)

    assert np.allclose(
        integral_result, integral_result_excepted, rtol=1e-8, atol=1e-12
    ), f"Expected {integral_result_excepted}, got {integral_result}"

    integral_result = double_integral(func2, ((0, 5), (0, 3)), 100, 100)
    print("integral result = ", integral_result)

    integral_result_excepted = 3.818641163

    assert np.allclose(
        integral_result, integral_result_excepted, rtol=1e-8, atol=1e-12
    ), f"Expected {integral_result_excepted}, got {integral_result}"

    integral_result = simple_integral(func3, (0, 5), 100)
    print("integral result = ", integral_result)
    integral_result_excepted = 17.99492253

    assert np.allclose(
        integral_result, integral_result_excepted, rtol=1e-8, atol=1e-12
    ), f"Expected {integral_result_excepted}, got {integral_result}"

    integral_result = simple_integral(func3, ([0, 5], [5, 0]), 100)

    print("integral result 2 = ", integral_result)
    integral_result_excepted = np.array([17.99492253, -17.99492253])

    assert np.allclose(
        integral_result, integral_result_excepted, rtol=1e-8, atol=1e-12
    ), f"Expected {integral_result_excepted}, got {integral_result}"

    h1 = np.random.rand(3, 3)
    h2 = np.random.rand(3, 3)

    integral_result1 = simple_integral(func3, (h1, h2), 100)

    integral_result2 = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            integral_result2[i, j] = simple_integral(func3, (h1[i, j], h2[i, j]), 100)

    assert np.allclose(
        integral_result1, integral_result2, rtol=1e-8, atol=1e-12
    ), f"Expected {integral_result1}, got {integral_result2}"

    print("integral result1 = \n", integral_result1)
    print("integral result2 = \n", integral_result2)


if __name__ == "__main__":
    test_integrals()
