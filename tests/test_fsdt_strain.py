import sympy as sym
import numpy as np

from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy import MidSurfaceGeometry, xi1_, xi2_
from shellpy.fsdt6.strain_vector import linear_strain_vector
from shellpy.fsdt_tensor.fosd_strain_tensor import fosd_linear_strain_components


def test_fsdt_strain():
    a = 1
    b = 1

    xi1 = 0.2
    xi2 = 0.3
    xi3 = 0.1

    U = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7])

    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    expansion_size = {"u1": (1, 1),
                      "u2": (1, 1),
                      "u3": (1, 1),
                      "v1": (1, 1),
                      "v2": (1, 1),
                      "v3": (1, 1)}

    boundary_conditions_u1 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_u2 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_u3 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_v1 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_v2 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}
    boundary_conditions_v3 = {"xi1": ("S", "S"),
                              "xi2": ("S", "S")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3,
                           "v1": boundary_conditions_v1,
                           "v2": boundary_conditions_v2,
                           "v3": boundary_conditions_v3}

    displacement_field = EigenFunctionExpansion(expansion_size, edges, boundary_conditions)

    R_ = sym.Matrix([xi1_, xi2_, (xi1_ - a / 2) ** 2 + (xi2_ - b / 2) ** 2 - (xi1_ - a / 2) * (xi2_ - b / 2)])

    mid_surface_geometry = MidSurfaceGeometry(R_)

    n_dof = displacement_field.number_of_degrees_of_freedom()

    epsilon0_lin = np.zeros((n_dof, 3, 3))
    epsilon1_lin = np.zeros((n_dof, 3, 3))
    epsilon2_lin = np.zeros((n_dof, 3, 3))

    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i], epsilon2_lin[i] = fosd_linear_strain_components(mid_surface_geometry,
                                                                                          displacement_field, i, xi1,
                                                                                          xi2)

    epsilon = epsilon0_lin + xi3 * epsilon1_lin + (xi3 ** 2) * epsilon2_lin


    epsilon = np.einsum('a, aij...->ij...', U, epsilon)

    epsilon_a = np.einsum('a, aij...->ij...', U, epsilon0_lin)
    print(epsilon_a)
    epsilon_b = np.einsum('a, aij...->ij...', U, epsilon1_lin)
    print(epsilon_b)
    epsilon_c = np.einsum('a, aij...->ij...', U, epsilon2_lin)
    print(epsilon_c)

    print('Strain tensor components test: \n', epsilon)

    MR1, MR2, MR3 = mid_surface_geometry.reciprocal_base(xi1, xi2)
    reciprocal_base = np.stack((MR1, MR2, MR3), axis=0)

    inverse_shift_tensor_extended = mid_surface_geometry.shifter_tensor_inverse_extended(xi1, xi2, xi3)

    MR = np.einsum('ba, bk->ak',
                   inverse_shift_tensor_extended,
                   reciprocal_base)

    E = np.einsum('ij,ia,jb->ab', epsilon, MR, MR)

    epsilon0_lin = np.zeros((n_dof, 6))
    epsilon1_lin = np.zeros((n_dof, 6))
    epsilon2_lin = np.zeros((n_dof, 6))

    for i in range(n_dof):
        epsilon0_lin[i], epsilon1_lin[i], epsilon2_lin[i] = linear_strain_vector(mid_surface_geometry,
                                                                                       displacement_field, i, xi1,
                                                                                       xi2)

    epsilon_voigt = epsilon0_lin + xi3 * epsilon1_lin + xi3 ** 2 * epsilon2_lin
    epsilon_voigt = np.einsum('a, ai->i', U, epsilon_voigt)

    print('Strain tensor components')
    print(epsilon_voigt)

    print('Strain tensor components (E=E_ij e_i x e_j)')
    print(E)

    E_expected = np.array([[0.357487815203159, 0.705352106783719, 0.401921276875195],
                           [0.705352106783719, -0.00774971001286451, 0.214157818495738],
                           [0.401921276875195, 0.214157818495738, -0.0884596633980486]])

    print('Strain tensor components expected (E=E_ij e_i x e_j)')
    print(E_expected)

    assert np.allclose(
        E, E_expected, rtol=1e-8, atol=1e-8
    ), f"Expected {E_expected}, got {E}"

    print('Strain tensor in curvilinear coordinates')
    print(epsilon)

    print('Strain vector in Voigt notation')
    print(epsilon_voigt)

    epsilon_voigt_expected = np.array(
        [epsilon[0, 0], epsilon[1, 1], epsilon[2, 2], 2 * epsilon[1, 2], 2 * epsilon[0, 2], 2 * epsilon[0, 1]])

    assert np.allclose(
        epsilon_voigt, epsilon_voigt_expected, rtol=1e-8, atol=1e-8
    ), f"Expected {epsilon_voigt_expected}, got {epsilon_voigt}"


if __name__ == "__main__":
    test_fsdt_strain()
