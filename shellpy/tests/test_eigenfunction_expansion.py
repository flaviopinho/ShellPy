import numpy as np
import matplotlib.pyplot as plt

from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy import RectangularMidSurfaceDomain

if __name__ == "__main__":
    a = 3
    b = 2

    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    expansion_size = {"u1": (2, 2),
                      "u2": (2, 2),
                      "u3": (3, 3)}

    boundary_conditions_u1 = {"xi1": ("F", "F"),
                              "xi2": ("F", "F")}
    boundary_conditions_u2 = {"xi1": ("F", "F"),
                              "xi2": ("F", "F")}
    boundary_conditions_u3 = {"xi1": ("S", "C"),
                              "xi2": ("C", "S")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3}

    expansion = EigenFunctionExpansion(expansion_size, edges, boundary_conditions)

    k = 1
    print("n_dof = ", expansion.number_of_degrees_of_freedom())
    trig_func = lambda _xi1, _xi2: expansion.shape_function(k, _xi1, _xi2)
    print("du = ", expansion.shape_function_first_derivatives(k, a / 4, b / 4))
    print("ddu = ", expansion.shape_function_second_derivatives(k, a / 4, b / 4))
    xi1 = np.linspace(*edges.edges["xi1"])
    xi2 = np.linspace(*edges.edges["xi2"])

    x, y = np.meshgrid(xi1, xi2, indexing='xy')

    du = expansion.shape_function_first_derivatives(k, x, y)
    ddu = expansion.shape_function_second_derivatives(k, x, y)

    print("du(grid) = ", np.shape(du))
    print("ddu(grid) = ", np.shape(ddu))

    z = trig_func(x, y)

    ax = plt.axes(projection='3d')
    # Creating plot
    ax.plot_surface(x, y, z[0]+z[1]+z[2])

    # show plot
    plt.show()
