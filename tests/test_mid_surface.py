import numpy as np
import sympy as sym

from shellpy import MidSurfaceGeometry, xi1_, xi2_

"""
Test script for the MidSurfaceGeometry class from shellpy.

This script performs a comprehensive verification of the midsurface geometry computations, 
including:

1. Evaluation of the midsurface position vector R(xi1, xi2).
2. Verification of the natural base vectors (M1, M2, M3) and reciprocal base vectors (M^1, M^2, M^3).
3. Computation of the metric tensor components (covariant and contravariant) and their extended forms.
4. Computation of the curvature tensor (covariant and mixed) and associated Christoffel symbols.
5. Verification of the shifter tensor and its inverse (including cubic approximation) for parallel surfaces.
6. Comparison of all computed quantities against expected reference values using np.allclose with strict tolerances.

The test is intended to ensure that the implementation of the midsurface geometry is 
consistent, accurate, and suitable for use in further shell analysis computations.
"""


def test_mid_surface():
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.01

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2) - xi1_])
    midsurface = MidSurfaceGeometry(R_)

    xi1 = 0.02
    xi2 = 0
    xi3 = 0.01

    # Resultado da função
    R_vector = midsurface(xi1, xi2)
    # Valor esperado
    R_vector_expected = np.array([[0.02],
                                  [0.0],
                                  [0.06124038405]])

    # Verificação
    print("Position vector")
    print("R(xi1,xi2) = \n", midsurface(xi1, xi2))
    assert np.allclose(R_vector, R_vector_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {R_vector_expected}, got {R_vector}"

    M1, M2, M3 = midsurface.natural_base(xi1, xi2)
    MR1, MR2, MR3 = midsurface.reciprocal_base(xi1, xi2)
    sqrtG = midsurface.sqrtG(xi1, xi2)

    print("\nNatural base")
    print("M_1(xi1,xi2) = \n", M1)
    print("M_2(xi1,xi2) = \n", M2)
    print("M_3(xi1,xi2) = \n", M3)

    M1_expected = np.array([1, 0, -0.6307255271])
    M2_expected = np.array([0, 1, 0.6154574550])
    M3_expected = np.array([0.4732005771, -0.4617457360, 0.7502480187])
    assert np.allclose(M1, M1_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {M1_expected}, got {M1}"
    assert np.allclose(M2, M2_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {M2_expected}, got {M2}"
    assert np.allclose(M3, M3_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {M3_expected}, got {M3}"

    print("\n sqrt(G) = ", sqrtG)
    sqrtG_expected = 1.332892557
    assert np.allclose(sqrtG, sqrtG_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {sqrtG_expected}, got {sqrtG}"

    print("\nReciprocal base")
    print("M^1(xi1,xi2) = \n", MR1)
    print("M^2(xi1,xi2) = \n", MR2)
    print("M^3(xi1,xi2) = \n", MR3)

    MR1_expected = np.array([0.7760812140, 0.2184983488, -0.3550177954])
    MR2_expected = np.array([0.2184983488, 0.7867908753, 0.3464238236])
    MR3_expected = np.array([0.4732005771, -0.4617457360, 0.7502480187])
    assert np.allclose(MR1, MR1_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {MR1_expected}, got {MR1}"
    assert np.allclose(MR2, MR2_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {MR2_expected}, got {MR2}"
    assert np.allclose(MR3, MR3_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {MR3_expected}, got {MR3}"

    print("\nMetric tensor")
    metric_tensor_covariant = midsurface.metric_tensor_covariant_components(xi1, xi2)
    metric_tensor_contravariant = midsurface.metric_tensor_contravariant_components(xi1, xi2)
    print("G_ab(xi1,xi2) = \n", metric_tensor_covariant)
    print("G^ab(xi1,xi2) = \n", metric_tensor_contravariant)

    metric_tensor_covariant_expected = np.array([[1.397814690, -0.3881847278], [-0.3881847278, 1.378787879]])
    metric_tensor_contravariant_expected = np.array([[0.7760812140, 0.2184983488], [0.2184983488, 0.7867908753]])

    assert np.allclose(metric_tensor_covariant, metric_tensor_covariant_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {metric_tensor_covariant_expected}, got {metric_tensor_covariant}"

    assert np.allclose(metric_tensor_contravariant, metric_tensor_contravariant_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {metric_tensor_contravariant_expected}, got {metric_tensor_contravariant}"

    print("\nMetric tensor extended")
    metric_tensor_covariant = midsurface.metric_tensor_covariant_components_extended(xi1, xi2)
    metric_tensor_contravariant = midsurface.metric_tensor_contravariant_components_extended(xi1, xi2)
    print("G_ij(xi1,xi2) = \n", metric_tensor_covariant)
    print("G^ij(xi1,xi2) = \n", metric_tensor_contravariant)

    metric_tensor_covariant_expected = np.array(
        [[1.397814690, -0.3881847278, 0], [-0.3881847278, 1.378787879, 0], [0, 0, 1]])
    metric_tensor_contravariant_expected = np.array(
        [[0.7760812140, 0.2184983488, 0], [0.2184983488, 0.7867908753, 0], [0, 0, 1]])

    assert np.allclose(metric_tensor_covariant, metric_tensor_covariant_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {metric_tensor_covariant_expected}, got {metric_tensor_covariant}"

    assert np.allclose(metric_tensor_contravariant, metric_tensor_contravariant_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {metric_tensor_contravariant_expected}, got {metric_tensor_contravariant}"

    print("\nCurvature tensor")
    K = midsurface.curvature_tensor_covariant_components(xi1, xi2)
    K_mixed = midsurface.curvature_tensor_mixed_components(xi1, xi2)
    H = midsurface.metric_tensor_contravariant_components(xi1, xi2)
    print("K_{alpha, beta}(xi1,xi2) = \n", K)
    print("K^{alpha}_{.beta}(xi1,xi2) = \n", K_mixed)

    K_expected = np.array([[10.49422128, 2.098844255], [2.098844255, 12.73298848]])
    K_mixed_expected = np.array([[8.602961995, 4.411010556], [3.944321530, 10.47679316]])

    assert np.allclose(K, K_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {K_expected}, got {K}"

    assert np.allclose(K_mixed, K_mixed_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {K_mixed_expected}, got {K}"

    print("\nChristoffel symbols")
    Gamma = midsurface.christoffel_symbols(xi1, xi2)
    print("C^{i}_{j alpha}(xi1,xi2) = \n", Gamma)

    Gamma1_expected = np.array([[4.965871563, 0.9931743125], [0.9931743125, 6.025257499], [8.602961995, 4.411010554]])
    Gamma2_expected = np.array(
        [[-4.845661927, -0.9691323852], [-0.9691323852, -5.879403140], [3.944321529, 10.47679315]])
    Gamma3_expected = np.array([[-10.49422128, -2.098844255], [-2.098844255, -12.73298849],
                                [0, 0]])
    assert np.allclose(Gamma[0], Gamma1_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {Gamma1_expected}, got {Gamma[0]}"

    assert np.allclose(Gamma[1], Gamma2_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {Gamma2_expected}, got {Gamma[1]}"

    assert np.allclose(Gamma[2], Gamma3_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {Gamma3_expected}, got {Gamma[2]}"

    print("\nChristoffel symbols derivative")
    dGamma = midsurface.christoffel_symbols_first_derivative(xi1, xi2)
    print("(C^{i}_{j alpha})_{,beta}(xi1,xi2) = \n", dGamma)

    dGamma_expected = np.array([[2.71766294, -32.56227785], [-32.56227785, 43.02440349], [-248.0057785, -104.0055818]])

    assert np.allclose(dGamma[0, :, :, 0], dGamma_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {dGamma_expected}, got {dGamma[0, :, :, 0]}"

    dGamma_expected = np.array(
        [[23.90540904, -28.32472860], [-28.32472860, -51.33153696], [-104.0055819, -157.2275769]])

    assert np.allclose(dGamma[0, :, :, 1], dGamma_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {dGamma_expected}, got {dGamma[0, :, :, 1]}"

    print("\nShifter tensor")
    shifter = midsurface.shifter_tensor(xi1, xi2, xi3)
    print("Upsilon^alpha_beta =  \n", shifter)

    shifter_expected = np.array([[1.086029620, 0.04411010556], [0.03944321530, 1.104767932]])

    assert np.allclose(shifter, shifter_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {shifter_expected}, got {shifter}"

    print("\nShifter tensor inverse")
    shifter_inv = midsurface.shifter_tensor_inverse(xi1, xi2, xi3)
    print("inv(Upsilon^alpha_beta).T = \n", np.linalg.inv(shifter.T))
    print("Lambda^alpha_beta = \n", shifter_inv)

    shifter_inv_expected = np.array(
        [[0.922122368009399, -0.0329222726699679], [-0.0368176101188025, 0.906481964144075]])

    assert np.allclose(shifter_inv, shifter_inv_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {shifter_inv_expected}, got {shifter_inv}"

    shifter_inv_approx = midsurface.shifter_tensor_inverse_approximation(xi1, xi2, xi3)
    print("Lambda^alpha_beta (cubic approx) = \n", shifter_inv_approx)

    shifter_inv_expected = np.array(
        [[0.922139863829303, -0.0329020538947510], [-0.0367949990739795, 0.906509065651631]])

    assert np.allclose(shifter_inv, shifter_inv_expected, rtol=1e-3,
                       atol=1e-3), f"Expected {shifter_inv_expected}, got {shifter_inv}"

    print("\nParallel surface basis\n")

    natural_base = np.stack((M1, M2, M3), axis=0)
    shift_tensor_extended = midsurface.shifter_tensor_extended(xi1, xi2, xi3)

    M = np.einsum('lj, lk->kj',
                  shift_tensor_extended,
                  natural_base)

    print("[M_1 M_2 M_3] (parallel surface)= \n", M)

    M1_expected = np.array([1.08602962, 0.0394432153, -0.6607109836151576])
    M2_expected = np.array([0.04411010556, 1.104767932, 0.6521162902145654])
    M3_expected = np.array([0.4732005771, -0.4617457360, 0.7502480187])

    M_expected = np.stack((M1_expected, M2_expected, M3_expected), axis=1)

    assert np.allclose(M, M_expected, rtol=1e-8,
                       atol=1e-10), f"Expected {M_expected}, got {M}"

    reciprocal_base = np.stack((MR1, MR2, MR3), axis=0)
    inverse_shift_tensor_extended = midsurface.shifter_tensor_inverse_extended(xi1, xi2, xi3)

    MR = np.einsum('lj, lk->kj',
                   inverse_shift_tensor_extended,
                   reciprocal_base)

    print("[M^1 M^2 M^3] (parallel surface)= \n", MR)

    MR1_expected = np.array([0.7075972597, 0.1725144551, -0.3401243474])
    MR2_expected = np.array([0.1725144549, 0.7060182758, 0.3257149406])
    MR3_expected = np.array([0.4732005771, -0.4617457360, 0.7502480187])

    MR_expected = np.stack((MR1_expected, MR2_expected, MR3_expected), axis=1)

    assert np.allclose(MR, MR_expected, rtol=1e-8,
                       atol=1e-10), f"Expected {MR_expected}, got {MR}"


if __name__ == "__main__":
    test_mid_surface()
