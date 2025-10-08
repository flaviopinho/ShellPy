import numpy as np

from shellpy import MidSurfaceGeometry, DisplacementExpansion, displacement_covariant_derivatives


def fosd2_linear_strain_vector(mid_surface_geometry: MidSurfaceGeometry,
                              displacement_expansion: DisplacementExpansion,
                              i: int, xi1, xi2):
    curvature_mixed = mid_surface_geometry.curvature_tensor_mixed_components(xi1, xi2)

    # shape functions e derivadas
    u, v = displacement_expansion.shape_function(i, xi1, xi2)
    du, dv = displacement_expansion.shape_function_first_derivatives(i, xi1, xi2)
    dcu, _, dcv, _ = displacement_covariant_derivatives(
        mid_surface_geometry, displacement_expansion, i, xi1, xi2
    )

    # inicializa tensores (3x3 simétricos)
    epsilon0 = np.zeros((3, 3) + np.shape(xi1))
    epsilon1 = np.zeros((3, 3) + np.shape(xi1))
    epsilon2 = np.zeros((3, 3) + np.shape(xi1))

    # --- epsilon0 ---
    epsilon0[:2, :2] = 0.5 * (dcu[:2] + np.moveaxis(dcu[:2], 0, 1))
    epsilon0[:2, 2] = 0.5 * (dcu[2, :2] + v[:2])
    epsilon0[2, :2] = epsilon0[:2, 2]
    epsilon0[2, 2] = v[2]

    # --- epsilon1 ---
    epsilon1[:2, :2] = 0.5 * (dcv[:2] + np.moveaxis(dcv[:2], 0, 1))
    epsilon1[:2, :2] += 0.5 * (
            np.einsum("oa...,ob...->ab...", curvature_mixed, dcu[:2])
            + np.einsum("ob...,oa...->ab...", curvature_mixed, dcu[:2])
    )
    epsilon1[:2, 2] = 0.5 * dv[2, :2]
    epsilon1[2, :2] = epsilon1[:2, 2]

    # --- epsilon2 ---
    epsilon2[:2, :2] = 0.5 * (
            np.einsum("oa...,ob...->ab...", curvature_mixed, dcv[:2])
            + np.einsum("ob...,oa...->ab...", curvature_mixed, dcv[:2])
    )

    # --- converter para Voigt (6 componentes independentes) ---
    def to_voigt(eps):
        return np.stack([
            eps[0, 0],  # ε11
            eps[1, 1],  # ε22
            eps[2, 2],  # ε33
            2*eps[1, 2],  # ε23
            2*eps[0, 2],  # ε13
            2*eps[0, 1],  # ε12
        ], axis=0)

    epsilon0 = to_voigt(epsilon0)
    epsilon1 = to_voigt(epsilon1)
    epsilon2 = to_voigt(epsilon2)

    return epsilon0, epsilon1, epsilon2
