import numpy as np

from shellpy import RectangularMidSurfaceDomain
from shellpy.numeric_integration.numeric_integration import simple_integral, double_integral


def func(xi1, xi2):
    return xi1 ** 2 + xi1 * xi2 + xi2 ** 2


def func2(xi1, xi2):
    return np.sin(11 * xi1 * np.pi / 5) * np.sin(9 * xi2 * np.pi / 3) * xi1 ** 3


def func3(xi3):
    return np.sin(11 * xi3 * np.pi / 5) * xi3 ** 3


if __name__ == "__main__":
    domain = RectangularMidSurfaceDomain(-2, 2, -2, 2)
    integral_result = double_integral(func, domain, 100, 100)
    print("integral result = ", integral_result)

    integral_result = double_integral(func, ((-2, 2), (-2, 2)), 100, 100)
    print("integral result = ", integral_result)

    integral_result = double_integral(func2, ((0, 5), (0, 3)), 100, 100)
    print("integral result = ", integral_result)

    integral_result = simple_integral(func3, (0, 5), 100)
    print("integral result = ", integral_result)
    # 17.99492253
