import numpy as np
import scienceplots
from shellpy import MidSurfaceGeometry, xi1_, xi2_
import sympy as sym
import matplotlib.pyplot as plt

# Given a spherical shell with radius R = 0.1 and
# projection dimension onto the horizontal plane of a=0.1 and b=0.1
# Center = a/2 and b/2

if __name__ == "__main__":
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.01

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
    xi1 = np.linspace(0, a, 3)
    xi2 = np.linspace(0, b, 3)
    xi3 = np.linspace(-h/2, h/2, 2)

    x, y = np.meshgrid(xi1, xi2, indexing='ij')
    z = xi3

    x0 = x[:, :]
    y0 = y[:, :]
    z0 = z[0]

    print("x0=", x0)
    print("y0=", y0)
    print("z0=", z0)

    RR = midsurface(x0, y0)
    print("R = ", np.shape(RR[0, 0]))

    M1, M2, M3 = midsurface.natural_base(x0, y0)
    MR1, MR2, MR3 = midsurface.reciprocal_base(x0, y0)
    print("M_1 = ", np.shape(M1))
    print("M^1 = ", np.shape(MR1))

    sqrtG = midsurface.sqrtG(x0, y0)
    print("sqrtG = ", np.shape(sqrtG))
    G = midsurface.metric_tensor_covariant_components(x0, y0)
    GR = midsurface.metric_tensor_covariant_components(x0, y0)
    print("G_ab= ", np.shape(G))
    print("G^ab= ", np.shape(GR))

    K = midsurface.curvature_tensor_covariant_components(x0, y0)
    KM = midsurface.curvature_tensor_mixed_components(x0, y0)
    print("K_ab= ", np.shape(K))
    print("K^a_b= ", np.shape(KM))

    C = midsurface.christoffel_symbols(x0, y0)
    print("C^i_(j alpha) = ", np.shape(C))

    dC = midsurface.christoffel_symbols_first_derivative(x0, y0)
    print("C^i_(j alpha, beta) = ", np.shape(dC))

    shifter = midsurface.shifter_tensor(x, y, z)
    print("Upsilon^alpha_beta = ", np.shape(shifter))

    print("I + K^a_b[:, :, 0, 0]*(-h / 2)= \n", np.eye(2, 2) + KM[:, :, 0, 0]*(-h / 2))
    print("Upsilon^alpha_beta[:, :, 0, 0, 0] = \n", shifter[:, :, 0, 0, 0])

    shifter_inv = midsurface.shifter_tensor_inverse(x, y, z)
    print("I - K^a_b[:, :, 0, 0]*(-h / 2)= \n", np.eye(2, 2) - KM[:, :, 0, 0] * (-h / 2))
    print("inv(Upsilon^alpha_beta[:, :, 0, 0, 0]) = \n", np.linalg.inv(shifter[:, :, 0, 0, 0]))
    print("Lambda^alpha_beta[:, :, 0, 0, 0] = \n", shifter_inv[:, :, 0, 0, 0])

    shifter_inv = midsurface.shifter_tensor_inverse_cubic_approximation(x, y, z)
    print("Lambda^alpha_beta[:, :, 0, 0, 0] (cubic approx) = \n", shifter_inv[:, :, 0, 0, 0])
