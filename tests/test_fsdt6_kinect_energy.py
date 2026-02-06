import sympy as sym
import numpy as np

from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion

from shellpy import RectangularMidSurfaceDomain, Shell, ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_
from shellpy.fsdt6.kinetic_energy import kinetic_energy
from shellpy.fsdt_tensor.fosd_kinetic_energy import fosd_kinetic_energy
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import (
    IsotropicHomogeneousLinearElasticMaterial,
)


def test_fosd_kinetic_energy():
    a = 1.0
    b = 1.0
    h = 0.1

    U = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7])

    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    expansion_size = {
        "u1": (1, 1),
        "u2": (1, 1),
        "u3": (1, 1),
        "v1": (1, 1),
        "v2": (1, 1),
        "v3": (1, 1),
    }

    boundary_conditions = {
        k: {"xi1": ("S", "S"), "xi2": ("S", "S")}
        for k in expansion_size
    }

    displacement_field = EigenFunctionExpansion(
        expansion_size, edges, boundary_conditions
    )

    R_ = sym.Matrix(
        [
            xi1_,
            xi2_,
            (xi1_ - a / 2) ** 2
            + (xi2_ - b / 2) ** 2
            - (xi1_ - a / 2) * (xi2_ - b / 2),
        ]
    )

    mid_surface_geometry = MidSurfaceGeometry(R_)

    material = IsotropicHomogeneousLinearElasticMaterial(
        E=10, nu=0.3, density=1550
    )

    shell = Shell(
        mid_surface_geometry,
        ConstantThickness(h),
        edges,
        material,
        displacement_field,
        None,
    )

    # --- FSDT kinetic energy ---
    T_fsdt_matrix = kinetic_energy(shell, 40, 40, 10)
    T_fsdt = np.einsum("ij,i,j->", T_fsdt_matrix, U, U)

    # --- FOSD kinetic energy ---
    T_fosd_matrix = fosd_kinetic_energy(shell, 40, 40, 10)
    T_fosd = np.einsum("ij,i,j->", T_fosd_matrix, U, U)

    # --- Comparison ---
    print(f"T (FSDT) = {T_fsdt}")
    print(f"T (FOSD) = {T_fosd}")

    assert np.isclose(
        T_fsdt,
        T_fosd,
        rtol=1e-6,
        atol=1e-8,
    ), f"Mismatch between kinetic energies: FSDT={T_fsdt}, FOSD={T_fosd}"


if __name__ == "__main__":
    test_fosd_kinetic_energy()
