import numpy as np

from shellpy.numeric_integration.boole_integral import boole_weights_simple_integral
from shellpy.numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_y
from shellpy.numeric_integration.integral_weights import double_integral_weights


def simple_integral(func, domain, n=n_integral_default_x, integral_rule=boole_weights_simple_integral):
    """
    Computes the numerical approximation of a single integral using a specified integration rule.

    Parameters:
        func (callable): The function to be integrated.
        domain (object): The domain over which the integration is performed.
        n (int, optional): Number of integration points. Defaults to n_integral_default_x.
        integral_rule (callable, optional): The numerical integration rule. Defaults to boole_weights_simple_integral.

    Returns:
        float: The approximated integral value.
    """
    x, W = integral_rule(domain, n)  # Obtain integration points and weights
    return np.sum(W * func(x))  # Compute the weighted sum of function values


def double_integral(func, domain, n_x=n_integral_default_x, n_y=n_integral_default_y,
                    integral_rule=boole_weights_simple_integral):
    """
    Computes the numerical approximation of a double integral using a specified integration rule.

    Parameters:
        func (callable): The function to be integrated, taking two arguments (x, y).
        domain (object): The domain over which the integration is performed.
        n_x (int, optional): Number of integration points in the x-direction. Defaults to n_integral_default_x.
        n_y (int, optional): Number of integration points in the y-direction. Defaults to n_integral_default_y.
        integral_rule (callable, optional): The numerical integration rule. Defaults to boole_weights_simple_integral.

    Returns:
        float: The approximated integral value.
    """
    x, y, Wxy = double_integral_weights(domain, n_x, n_y, integral_rule)  # Obtain integration points and weights
    return np.sum(Wxy * func(x, y))  # Compute the weighted sum of function values

