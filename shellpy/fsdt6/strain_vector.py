import numpy as np

from shellpy import MidSurfaceGeometry, DisplacementExpansion, displacement_first_covariant_derivatives


def linear_strain_vector(mid_surface_geometry: MidSurfaceGeometry,
                         displacement_expansion: DisplacementExpansion,
                         i: int, xi1, xi2):
    curvature_mixed = mid_surface_geometry.curvature_tensor_mixed_components(xi1, xi2)

    # shape functions e derivadas
    U = displacement_expansion.shape_function(i, xi1, xi2)
    dU = displacement_expansion.shape_function_first_derivatives(i, xi1, xi2)
    u = U[0:3]
    v = U[3:6]
    du = dU[0:3]
    dv = dU[3:6]
    dcu = displacement_first_covariant_derivatives(
        mid_surface_geometry, u, du, xi1, xi2
    )
    dcv = displacement_first_covariant_derivatives(
        mid_surface_geometry, v, dv, xi1, xi2
    )

    # --- Inicializa tensores (3x3 simétricos) ---
    epsilon0 = np.zeros((3, 3) + np.shape(xi1))
    epsilon1 = np.zeros((3, 3) + np.shape(xi1))
    epsilon2 = np.zeros((3, 3) + np.shape(xi1))

    # --- epsilon0 ---
    # Parte métrica linear (membrana)
    aux = np.swapaxes(dcu, 0, 1)
    epsilon0[0:2, 0:2] = 0.5 * (dcu[0:2, 0:2] + aux[0:2, 0:2])

    # Cisalhamento transversal (epsilon_13 e epsilon_23)
    epsilon0[0:2, 2] = 0.5 * (dcu[2, 0:2] + v[0:2])
    epsilon0[2, 0:2] = epsilon0[0:2, 2]

    # Termo normal
    epsilon0[2, 2] = v[2]

    # --- epsilon1 ---
    # Derivadas associadas às rotações (efeito de espessura linear)
    aux = np.swapaxes(dcv, 0, 1)
    epsilon1[0:2, 0:2] = 0.5 * (dcv[0:2, 0:2] + aux[0:2, 0:2])

    # Correções devido à curvatura da superfície média
    upsilon_dcu = 0.5 * (
            np.einsum('oa..., ob...->ab...', curvature_mixed, dcu[0:2, 0:2])
            + np.einsum('ob..., oa...->ab...', curvature_mixed, dcu[0:2, 0:2])
    )
    epsilon1[0:2, 0:2] += upsilon_dcu

    # Cisalhamento transversal linear com espessura
    epsilon1[0:2, 2] = 0.5 * dv[2, 0:2]
    epsilon1[2, 0:2] = epsilon1[0:2, 2]

    # --- epsilon2 ---
    # Termos de segunda ordem em relação à espessura
    upsilon_dcv = 0.5 * (
            np.einsum('oa..., ob...->ab...', curvature_mixed, dcv[0:2, 0:2])
            + np.einsum('ob..., oa...->ab...', curvature_mixed, dcv[0:2, 0:2])
    )
    epsilon2[0:2, 0:2] = upsilon_dcv

    # --- converter para Voigt (6 componentes independentes) ---

    epsilon0 = to_voigt(epsilon0)
    epsilon1 = to_voigt(epsilon1)
    epsilon2 = to_voigt(epsilon2)

    return epsilon0, epsilon1, epsilon2


def nonlinear_strain_vector(mid_surface_geometry: MidSurfaceGeometry,
                                displacement_expansion: DisplacementExpansion,
                                i: int, j: int, xi1, xi2):
    metric_tensor_contravariant_components = mid_surface_geometry.metric_tensor_contravariant_components_extended(xi1,
                                                                                                                  xi2)

    # shape functions e derivadas
    U_I = displacement_expansion.shape_function(i, xi1, xi2)
    dU_I = displacement_expansion.shape_function_first_derivatives(i, xi1, xi2)
    u_I = U_I[0:3]
    v_I = U_I[3:6]
    du_I = dU_I[0:3]
    dv_I = dU_I[3:6]
    dcu_I = displacement_first_covariant_derivatives(
        mid_surface_geometry, u_I, du_I, xi1, xi2
    )
    dcv_I = displacement_first_covariant_derivatives(
        mid_surface_geometry, v_I, dv_I, xi1, xi2
    )

    # shape functions e derivadas
    U_J = displacement_expansion.shape_function(j, xi1, xi2)
    dU_J = displacement_expansion.shape_function_first_derivatives(j, xi1, xi2)
    u_J = U_J[0:3]
    v_J = U_J[3:6]
    du_J = dU_J[0:3]
    dv_J = dU_J[3:6]
    dcu_J = displacement_first_covariant_derivatives(
        mid_surface_geometry, u_J, du_J, xi1, xi2
    )
    dcv_J = displacement_first_covariant_derivatives(
        mid_surface_geometry, v_J, dv_J, xi1, xi2
    )

    epsilon0 = np.zeros((3, 3) + np.shape(xi1))
    epsilon1 = np.zeros((3, 3) + np.shape(xi1))
    epsilon2 = np.zeros((3, 3) + np.shape(xi1))

    epsilon0[0:2, 0:2] = 1 / 2 * np.einsum('oa..., ot..., tb... -> ab...',
                                           dcu_I,
                                           metric_tensor_contravariant_components,
                                           dcu_J)

    epsilon1[0:2, 0:2] = 1 / 2 * np.einsum('oa..., ot..., tb... -> ab...',
                                           dcu_I,
                                           metric_tensor_contravariant_components,
                                           dcv_J)
    epsilon1[0:2, 0:2] += 1 / 2 * np.einsum('oa..., ot..., tb... -> ab...',
                                            dcv_I,
                                            metric_tensor_contravariant_components,
                                            dcu_J)

    epsilon2[0:2, 0:2] = 1 / 2 * np.einsum('oa..., ot..., tb... -> ab...',
                                           dcv_I,
                                           metric_tensor_contravariant_components,
                                           dcv_J)

    epsilon0[2, 0:2] = 1 / 2 * np.einsum('t...,ot...,ob...->b...',
                                         v_I,
                                         metric_tensor_contravariant_components,
                                         dcu_J)

    epsilon0[0:2, 2] = 1 / 2 * np.einsum('ob...,ot...,t...->b...',
                                         dcu_I,
                                         metric_tensor_contravariant_components,
                                         v_J)

    epsilon1[2, 0:2] = 1 / 2 * np.einsum('t...,ot...,ob...->b...',
                                         v_I,
                                         metric_tensor_contravariant_components,
                                         dcv_J)

    epsilon1[0:2, 2] = 1 / 2 * np.einsum('ob...,ot...,t...->b...',
                                         dcv_I,
                                         metric_tensor_contravariant_components,
                                         v_J)

    epsilon0[2, 2] = 1 / 2 * np.einsum('o...,ot...,t...->...',
                                       v_I,
                                       metric_tensor_contravariant_components,
                                       v_J)

    epsilon0 = to_voigt(epsilon0)
    epsilon1 = to_voigt(epsilon1)
    epsilon2 = to_voigt(epsilon2)

    return epsilon0, epsilon1, epsilon2

def to_voigt(eps):
    return np.stack([
        eps[0, 0],  # ε11
        eps[1, 1],  # ε22
        eps[2, 2],  # ε33
        (eps[1, 2] + eps[2, 1]),  # g23
        (eps[0, 2] + eps[2, 0]),  # g13
        (eps[0, 1] + eps[1, 0]),  # g12
    ], axis=0)
