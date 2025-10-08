import sympy as sym
import numpy as np

from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy.koiter_shell_theory.koiter_strain_tensor import (
    koiter_linear_strain_components,
    koiter_nonlinear_strain_components_total,
)
from shellpy import RectangularMidSurfaceDomain, MidSurfaceGeometry, xi1_, xi2_


def test_koiter_strain_components():
    """
    Test for the linear and nonlinear strain components
    from Koiter’s shell theory applied to a spherical mid-surface.
    """

    # --- Geometric and mechanical parameters ---
    R = 0.1       # radius of the mid-surface (m)
    a = 0.1       # length in xi1-direction (m)
    b = 0.1       # length in xi2-direction (m)
    h = 0.0001    # shell thickness (m)
    E = 10        # Young’s modulus (Pa)
    nu = 0.2      # Poisson’s ratio

    # --- Definition of rectangular mid-surface domain ---
    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    # --- Expansion size: number of modes for each displacement component ---
    expansion_size = {"u1": (1, 1), "u2": (1, 1), "u3": (1, 1)}

    # --- Boundary conditions: simply supported (S) on all sides ---
    boundary_conditions_u1 = {"xi1": ("S", "S"), "xi2": ("S", "S")}
    boundary_conditions_u2 = {"xi1": ("S", "S"), "xi2": ("S", "S")}
    boundary_conditions_u3 = {"xi1": ("S", "S"), "xi2": ("S", "S")}

    boundary_conditions = {
        "u1": boundary_conditions_u1,
        "u2": boundary_conditions_u2,
        "u3": boundary_conditions_u3,
    }

    # --- Creation of modal displacement field using eigenfunction expansion ---
    displacement_field = EigenFunctionExpansion(expansion_size, edges, boundary_conditions)

    # --- Symbolic definition of mid-surface geometry ---
    # R_ represents the position vector of a point on the mid-surface
    R_ = sym.Matrix([
        xi1_,
        xi2_,
        sym.sqrt(R**2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)
    ])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    # ===============================================================
    #                     LINEAR STRAIN COMPONENTS
    # ===============================================================
    print("\nLinear strain components")
    gamma, rho = koiter_linear_strain_components(
        mid_surface_geometry, displacement_field, 1, a / 4, b / 3
    )

    # Display computed results
    print("gamma_lin = \n", gamma)
    print("rho_lin = \n", rho)

    # Expected reference values (analytical or benchmark)
    gamma_expected = np.array([[1.090739709, 9.665869714],
                               [9.665869714, 12.15899207]])
    rho_expected = np.array([[21.81479417, 193.3173943],
                             [193.3173942, 243.1798415]])

    # --- Numerical validation of linear strain components ---
    assert np.allclose(
        gamma, gamma_expected, rtol=1e-8, atol=1e-12
    ), f"Expected {gamma_expected}, got {gamma}"

    assert np.allclose(
        rho, rho_expected, rtol=1e-8, atol=1e-12
    ), f"Expected {rho_expected}, got {rho}"

    # ===============================================================
    #                     NONLINEAR STRAIN COMPONENTS
    # ===============================================================
    print("\nNonlinear strain components")
    gamma_nl = koiter_nonlinear_strain_components_total(
        mid_surface_geometry, displacement_field, 1, 0, a / 4, b / 3
    )

    # Display computed results
    print("gamma_nonlin = \n", gamma_nl)

    # Expected reference values for nonlinear case
    gamma_nl_expected = np.array([[2.941772279, 15.97838764],
                                  [14.33404904, 6.737080007]])

    # --- Numerical validation of nonlinear strain components ---
    assert np.allclose(
        gamma_nl, gamma_nl_expected, rtol=1e-8, atol=1e-12
    ), f"Expected {gamma_nl_expected}, got {gamma_nl}"


if __name__ == "__main__":
    # Allows the test to be executed directly (python this_file.py)
    test_koiter_strain_components()
