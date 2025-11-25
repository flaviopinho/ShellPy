import sympy as sym
import numpy as np

from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy.fosd_theory.fosd_strain_tensor import fosd_linear_strain_components
from shellpy.fosd_theory2.fosd2_strain_vector import fosd2_linear_strain_vector
from shellpy import RectangularMidSurfaceDomain, displacement_first_covariant_derivatives
from shellpy import MidSurfaceGeometry, xi1_, xi2_


def test_covariant_derivative():

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

    u_aux = np.zeros((6,))
    du_aux = np.zeros((6, 2))
    for i in range(n_dof):
        u_aux += displacement_field.shape_function(i, xi1, xi2) * U[i]
        du_aux += displacement_field.shape_function_first_derivatives(i, xi1, xi2) * U[i]

    u_t = u_aux[0:3] + xi3 * u_aux[3:6]
    du_t = du_aux[0:3] + xi3 * du_aux[3:6]

    dcu=np.zeros((3,3))
    dcu[:,0:2] = displacement_first_covariant_derivatives(
        mid_surface_geometry, u_t, du_t, xi1, xi2
    )
    K = mid_surface_geometry.curvature_tensor_mixed_components(xi1, xi2)

    aux = dcu[2, 0:2]+np.einsum('oa, o->a', K, u_t[0:2])
    print(aux)
    print(du_t[2, 0:2])



    M1, M2, M3 = mid_surface_geometry.natural_base(xi1, xi2)
    MR1, MR2, MR3 = mid_surface_geometry.reciprocal_base(xi1, xi2)

    du_t_1 = dcu[0, 0] * MR1 + dcu[1, 0] * MR2 + dcu[2, 0] * MR3
    du_t_2 = dcu[0, 1] * MR1 + dcu[1, 1] * MR2 + dcu[2, 1] * MR3
    du_t_3 = u_aux[3] * MR1 + u_aux[4] * MR2 + u_aux[5] * MR3
    dcu[:, 2] = u_aux[3:6]

    print("\nDerivative of vector U")
    print("U_(,1)(xi1,xi2,xi3) = \n", du_t_1)
    print("U_(,2)(xi1,xi2,xi3) = \n", du_t_2)
    print("U_(,3)(xi1,xi2,xi3) = \n", du_t_3)

    """
    du_t_1_expected = np.array([0.30259985411741661746, 0.72471666829870692459, 0.62396633795955680006])

    du_t_2_expected = np.array([0.46092506980733540335, 0.06938772409201797913, 0.42601940680345777766])
    assert np.allclose(du_t_1, du_t_1_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {du_t_1_expected}, got {du_t_2}"
    assert np.allclose(du_t_2, du_t_2_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {du_t_2_expected}, got {du_t_2}"

    """
    print("\nCovariant derivative components")
    print(dcu)

    dcu_expected = np.array(
        [[0.05301331893359389744, 0.29051730708595229229], [0.66232003450275124458, 0.026785783411672201364],
         [0.75575862883010861809, 0.57072012938913789088]])
    """
    assert np.allclose(dcu[:,0:2], dcu_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {dcu_expected}, got {dcu[:, 0:2]}"
                       
    """

    upsilon = mid_surface_geometry.shifter_tensor_extended(xi1, xi2, xi3)

    dcu_P = upsilon @ dcu

    print(dcu_P)


if __name__ == "__main__":
    test_covariant_derivative()
