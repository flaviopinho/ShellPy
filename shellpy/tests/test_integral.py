import numpy as np

from shellpy import double_integral_booles_rule
from shellpy import RectangularMidSurfaceDomain


def func(xi1, xi2):
    return xi1 ** 2 + xi1 * xi2 + xi2 ** 2

def func2(xi1, xi2):
    return np.sin(11*xi1*np.pi/5) *np.sin(9*xi2*np.pi/3)*xi1**3


if __name__ == "__main__":
    rect = RectangularMidSurfaceDomain(-2, 2, -2, 2);
    integral_result = double_integral_booles_rule(func, rect, 10)
    print("integral result = ", integral_result)

    rect = RectangularMidSurfaceDomain(0, 5, 0, 3)
    integral_result = double_integral_booles_rule(func2, rect, 100)
    print("integral result = ", integral_result)
