import numpy as np
from shellpy.multiindex import MultiIndex


def test_multiindex_notation():
    # functional u
    # u = 10 x1^2*x2*x3 - 5 x2 + 7*x1*x3^3
    # x = [ -5, 3, 2 ]
    # u(x) = 1205

    functional_u = MultiIndex(3)
    functional_u.add_monomial([2, 1, 1], 10)
    functional_u.add_monomial([0, 1, 0], -5)
    functional_u.add_monomial([1, 0, 3], 7)
    x = np.array([-5, 3, 2])
    result = functional_u(x)
    print("u(x) = ", functional_u(x))
    result_expected = 1205

    assert np.allclose(result, result_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {result_expected}, got {result}"

    # Jacobian of functional u
    # J = Jacobian(U)
    # J(x) = [-544, 495, 330]
    jacobian_u = functional_u.jacobian()
    print("jacob(u)(x) = ", jacobian_u(x))

    jacobian_u_expected = np.array([-544, 495, 330])

    assert np.allclose(jacobian_u(x), jacobian_u_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {jacobian_u_expected}, got {jacobian_u(x)}"

    # Hessian of functional
    hessian_u = jacobian_u.jacobian()
    print("hessian(u)(x) = ", hessian_u(x))

    hessian_u_expected = np.array([[120, -200, -216], [-200, 0, 250], [-216, 250, -420]])
    assert np.allclose(hessian_u(x), hessian_u_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {hessian_u_expected}, got {hessian_u(x)}"

    # A=[[2 * x1 ^ 2,                 4 * x2,                    -x3],
    #  [-12 * x3 ^ 3,        -15 * x1 * x3,            x2 * x1 ^ 2],
    #  [8 * x3 ^ 2 * x2, -5 * x1 * x3 * x2, -10 * x3 * x2 * x1 ^ 2]])

    A = MultiIndex(3, (3, 3))  # Matriz 3x3
    A.add_monomial((2, 0, 0), 2.0, (0, 0))
    A.add_monomial((0, 1, 0), 4.0, (0, 1))
    A.add_monomial((0, 0, 1), -1.0, (0, 2))

    A.add_monomial((0, 0, 3), -12.0, (1, 0))
    A.add_monomial((1, 0, 1), -15.0, (1, 1))
    A.add_monomial((2, 1, 0), 1.0, (1, 2))

    A.add_monomial((0, 1, 2), 8, (2, 0))
    A.add_monomial((1, 1, 1), -5, (2, 1))
    A.add_monomial((2, 1, 1), -10.0, (2, 2))

    x = np.array([10, -2, 5])

    print("A(x) = ", A(x))

    A_expected = np.array([[200, -8, -5], [-1500, -750, -200], [-400, 500, 10000]])

    assert np.allclose(A(x), A_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {A_expected}, got {A(x)}"

    B = MultiIndex(3, (3,))  # Matriz 3x3
    B.add_monomial((2, 0, 0), 10.0, (0,))
    B.add_monomial((1, 3, 0), 2.0, (1,))
    B.add_monomial((0, 0, 1), -5.0, (2,))

    print("B(x) = ", B(x))

    B_expected = np.array([1000, -160, -25])

    assert np.allclose(B(x), B_expected, rtol=1e-8,
                       atol=1e-12), f"Expected {B_expected}, got {B(x)}"


if __name__ == "__main__":
    test_multiindex_notation()
