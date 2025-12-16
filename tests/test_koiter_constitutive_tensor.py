import numpy as np
from shellpy import MidSurfaceGeometry, xi1_, xi2_
import sympy as sym

from shellpy.koiter_shell_theory.constitutive_tensor_koiter import plane_stress_constitutive_tensor_for_koiter_theory
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial

def test_koiter_constitutive_tensor():
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.001

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2) - xi1_])
    mid_surface = MidSurfaceGeometry(R_)

    material = IsotropicHomogeneousLinearElasticMaterial(10, 0.3, 0)

    xi1 = 0.02
    xi2 = 0
    xi3 = 0.01

    C = plane_stress_constitutive_tensor_for_koiter_theory(mid_surface, material, xi1, xi2, xi3)
    C = C.reshape(4, 4)
    print(C)

    C_expected = np.array([[6.618703859631808, 1.863433669001657, 1.863433669001657, 2.380254453134639],
                           [1.863433669001657, 2.6895248322620158, 2.6895248322620158, 1.8891484296048946],
                           [1.863433669001657, 2.6895248322620158, 2.6895248322620158, 1.8891484296048946],
                           [2.380254453134639, 1.8891484296048948, 1.8891484296048948, 6.8026360579870095]])
    print(C_expected)

    assert np.allclose(
        C, C_expected, rtol=1e-8, atol=1e-8
    ), f"Expected {C_expected}, got {C}"

def test_koiter_constitutive_tensor_grid():
    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.001

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2) - xi1_])
    mid_surface = MidSurfaceGeometry(R_)

    material = IsotropicHomogeneousLinearElasticMaterial(10, 0.3, 0)

    xi1_lin = np.linspace(0, a, 4)
    xi2_lin = np.linspace(0, b, 6)

    xi1, xi2 = np.meshgrid(xi1_lin, xi2_lin, indexing='ij')

    xi3 = np.linspace(-h / 2, h / 2, 2)

    C_grid = plane_stress_constitutive_tensor_for_koiter_theory(mid_surface, material, xi1, xi2, xi3)

    for i in range(xi1.shape[0]):
        for j in range(xi2.shape[1]):
            for k in range(xi3.shape[-1]):
                x = xi1[i, j]
                y = xi2[i, j]
                z = xi3[k]
                C = plane_stress_constitutive_tensor_for_koiter_theory(mid_surface, material, x, y, z)

                assert np.allclose(C_grid[:, :, :, :, i, j, k], C, rtol=1e-8,
                                   atol=1e-12), f"Expected {C}, got {C_grid[:, :, :, :, i, j, k]}"


if __name__ == "__main__":
    test_koiter_constitutive_tensor()
    test_koiter_constitutive_tensor_grid()

