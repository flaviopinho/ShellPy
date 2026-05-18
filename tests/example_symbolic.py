import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import inspect

if __name__ == "__main__":

    R = 1
    a = 0.1
    b = 0.1

    # Position vector
    Xi1, Xi2 = sym.symbols('Xi1 Xi2')
    R = sym.Matrix([Xi1, Xi2, sym.sqrt(R ** 2 - (Xi1 - a / 2) ** 2 - (Xi2 - b / 2) ** 2)])

    lambda_R = sym.lambdify((Xi1, Xi2), R, "numpy")
    print("R(a/2, b/2) = {}".format(lambda_R(a / 2, b / 2)))

    # Natural base

    M1 = sym.diff(R, Xi1, 1)
    M2 = sym.diff(R, Xi2, 1)
    Maux = M1.cross(M2)
    sqrtG = sym.sympify(sym.sqrt(Maux.dot(Maux)))
    M3 = sym.simplify(Maux / sqrtG)

    print("M1 = ", M1)
    print("M2 = ", M2)
    print("M3 = ", M3)

    # Metric tensor
    G = sym.Matrix([[M1.dot(M1), M1.dot(M2)], [M2.dot(M1), M2.dot(M2)]])
    print("G = ", G)

    # Natural base derivative
    M1_1 = sym.diff(M1, Xi1, 1)
    M1_2 = sym.diff(M1, Xi2, 1)
    M2_1 = sym.diff(M2, Xi1, 1)
    M2_2 = sym.diff(M2, Xi2, 1)
    M3_1 = sym.diff(M3, Xi1, 1)
    M3_2 = sym.diff(M3, Xi2, 1)

    print("M3_2 = ", M3_2)

    xi1 = np.linspace(0, a)
    xi2 = np.linspace(0, b)

    x, y = np.meshgrid(xi1, xi2, indexing='xy')

    z = lambda_R(x, y)

    print(np.shape(z))

    A = sym.tensor.array.MutableDenseNDimArray.zeros(2,2)
    print(A)
    A[0, 0] = Xi1 ** 2
    B=sym.Array(A, (2,2))
    M = sym.lambdify(Xi1, B, "numpy")
    print(inspect.getsource(M))

    ax = plt.axes(projection='3d')

    # Creating plot
    ax.plot_surface(x, y, z[2, 0])

    # show plot
    plt.show()

