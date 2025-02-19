import numpy as np
import scienceplots
from midsurface_geometry import MidSurfaceGeometry, xi1_, xi2_
import sympy as sym
import matplotlib.pyplot as plt

# Given a spherical shell with radius R = 0.1 and
# projection dimension onto the horizontal plane of a=0.1 and b=0.1
# Center = a/2 and b/2

if __name__ == "__main__":
    R = 0.1
    a = 0.1
    b = 0.1

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    midsurface = MidSurfaceGeometry(R_)

    M1, M2, M3 = midsurface.natural_base(a / 4, b / 3)
    MR1, MR2, MR3 = midsurface.reciprocal_base(a / 4, b / 3)
    sqrtG = midsurface.sqrtG(a / 4, b / 3)

    print("Position vector")
    print("R(a/4,b/3) = \n", midsurface(a / 4, b / 3))
    print("\nNatural base")
    print("M_1(a/4,b/3) = \n", M1)
    print("M_2(a/4,b/3) = \n", M2)
    print("M_3(a/4,b/3) = \n", M3)

    print("\n sqrt(G) = ", sqrtG)

    print("\nReciprocal base")
    print("M^1(a/4,b/3) = \n", MR1)
    print("M^2(a/4,b/3) = \n", MR2)
    print("M^3(a/4,b/3) = \n", MR3)

    print("\nMetric tensor")
    print("G_ab(a/4,b/3) = \n", midsurface.metric_tensor_covariant_components(a / 4, b / 3))
    print("G^ab(a/4,b/3) = \n", midsurface.metric_tensor_contravariant_components(a / 4, b / 3))

    print("\nCurvature tensor")
    print("K_{alpha, beta}(a/4,b/3) = \n", midsurface.curvature_tensor_covariant_components(a / 4, b / 3))
    print("K^{alpha}_{.beta}(a/4,b/3) = \n", midsurface.curvature_tensor_mixed_components(a / 4, b / 3))
    H = midsurface.metric_tensor_contravariant_components(a / 4, b / 3)
    K = midsurface.curvature_tensor_covariant_components(a / 4, b / 3)
    print(H @ K)

    print("\nChristoffel symbols")
    print("C^{i}_{j alpha}(a/4,b/3) = \n", midsurface.christoffel_symbols(a / 4, b / 3))

    print("\nChristoffel symbols derivative")
    print("(C^{i}_{j alpha})_{,beta}(a/4,b/3) = \n", midsurface.christoffel_symbols_first_derivative(a / 4, b / 3))

    # Test grid
    xi1 = np.linspace(0, a, 100)
    xi2 = np.linspace(0, b, 100)

    x, y = np.meshgrid(xi1, xi2)

    RR = midsurface(x, y)
    print("R = ", np.shape(RR[0, 0]))

    M1, M2, M3 = midsurface.natural_base(x, y)
    MR1, MR2, MR3 = midsurface.reciprocal_base(x, y)
    print("M_1 = ", np.shape(M1))
    print("M^1 = ", np.shape(MR1))

    sqrtG = midsurface.sqrtG(x, y)
    print("sqrtG = ", np.shape(sqrtG))
    G = midsurface.metric_tensor_covariant_components(x, y)
    GR = midsurface.metric_tensor_covariant_components(x, y)
    print("G_ab= ", np.shape(G))
    print("G^ab= ", np.shape(GR))

    K = midsurface.curvature_tensor_covariant_components(x, y)
    KM = midsurface.curvature_tensor_mixed_components(x, y)
    print("K_ab= ", np.shape(K))
    print("K^a_b= ", np.shape(KM))

    C = midsurface.christoffel_symbols(x, y)
    print("C^i_(j alpha) = ", np.shape(C))

    dC = midsurface.christoffel_symbols_first_derivative(x, y)
    print("C^i_(j alpha, beta) = ", np.shape(dC))

    plt.style.use('science')

    ax = plt.axes(projection='3d')

    # Creating plot
    ax.plot_surface(RR[0, 0], RR[1, 0], RR[2, 0])

    # show plot
    plt.show()
