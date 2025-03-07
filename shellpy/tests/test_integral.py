import numpy as np

from shellpy import double_integral_booles_rule, boole_weights_simple_integral


def func(xi1, xi2):
    return xi1 ** 2 + xi1 * xi2 + xi2 ** 2

def func2(xi1, xi2):
    return np.sin(11*xi1*np.pi/5) *np.sin(9*xi2*np.pi/3)*xi1**3

def func3(xi3):
    return np.sin(11*xi3*np.pi/5)*xi3**3


if __name__ == "__main__":
    integral_result = double_integral_booles_rule(func, (-2, 2), (-2, 2), 10, 10)
    print("integral result = ", integral_result)

    integral_result = double_integral_booles_rule(func2, (0, 5), (0, 3), 100, 100)
    print("integral result = ", integral_result)

    z, Wz = boole_weights_simple_integral((0, 5))

    print("integral result = ", np.dot(func3(z), Wz))
    #17.99492253
