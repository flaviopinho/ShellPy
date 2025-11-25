import numpy as np

from shellpy import MidSurfaceGeometry, DisplacementExpansion, displacement_first_covariant_derivatives


def fosd_linear_strain_components(mid_surface_geometry: MidSurfaceGeometry,
                                  displacement_expansion: DisplacementExpansion,
                                  i: int, xi1, xi2):
    curvature_mixed = mid_surface_geometry.curvature_tensor_mixed_components(xi1, xi2)

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

    return epsilon0, epsilon1, epsilon2


def fosd_nonlinear_strain_components(mid_surface_geometry: MidSurfaceGeometry,
                                     displacement_expansion: DisplacementExpansion,
                                     i: int, j: int, xi1, xi2):
    metric_tensor_contravariant_components = mid_surface_geometry.metric_tensor_contravariant_components_extended(xi1, xi2)

    # u_i: The displacement associated with the shape function i at coordinates (xi1, xi2)
    uI, vI = displacement_expansion.shape_function(i, xi1, xi2)

    # u_{i,alpha}: The first derivatives of the displacement u with respect to the curvilinear coordinates
    duI, dvI = displacement_expansion.shape_function_first_derivatives(i, xi1, xi2)

    # dcu: displacement covariant derivatives
    # ddcu: second covariant derivatives of displacement
    dcuI, _, dcvI, _ = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, i, xi1, xi2)

    # u_i: The displacement associated with the shape function i at coordinates (xi1, xi2)
    uJ, vJ = displacement_expansion.shape_function(j, xi1, xi2)

    # u_{i,alpha}: The first derivatives of the displacement u with respect to the curvilinear coordinates
    duJ, dvJ = displacement_expansion.shape_function_first_derivatives(j, xi1, xi2)

    # dcu: displacement covariant derivatives
    # ddcu: second covariant derivatives of displacement
    dcuJ, _, dcvJ, _ = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, j, xi1, xi2)

    epsilon0 = np.zeros((3, 3) + np.shape(xi1))
    epsilon1 = np.zeros((3, 3) + np.shape(xi1))
    epsilon2 = np.zeros((3, 3) + np.shape(xi1))

    epsilon0[0:2, 0:2] = 1 / 2 * np.einsum('oa..., ot..., tb... -> ab...',
                                           dcuI,
                                           metric_tensor_contravariant_components,
                                           dcuJ)

    epsilon1[0:2, 0:2] = 1 / 2 * np.einsum('oa..., ot..., tb... -> ab...',
                                           dcuI,
                                           metric_tensor_contravariant_components,
                                           dcvJ)
    epsilon1[0:2, 0:2] += 1 / 2 * np.einsum('oa..., ot..., tb... -> ab...',
                                            dcvI,
                                            metric_tensor_contravariant_components,
                                            dcuJ)

    epsilon2[0:2, 0:2] = 1 / 2 * np.einsum('oa..., ot..., tb... -> ab...',
                                           dcvI,
                                           metric_tensor_contravariant_components,
                                           dcvJ)

    epsilon0[2, 0:2] = 1 / 2 * np.einsum('t...,ot...,ob...->b...',
                                         vI,
                                         metric_tensor_contravariant_components,
                                         dcuJ)

    epsilon0[0:2, 2] = 1 / 2 * np.einsum('ob...,ot...,t...->b...',
                                         dcuI,
                                         metric_tensor_contravariant_components,
                                         vJ)

    epsilon1[2, 0:2] = 1 / 2 * np.einsum('t...,ot...,ob...->b...',
                                         vI,
                                         metric_tensor_contravariant_components,
                                         dcvJ)

    epsilon1[0:2, 2] = 1 / 2 * np.einsum('ob...,ot...,t...->b...',
                                         dcvI,
                                         metric_tensor_contravariant_components,
                                         vJ)

    epsilon0[2, 2] = 1 / 2 * np.einsum('o...,ot...,t...->...',
                                       vI,
                                       metric_tensor_contravariant_components,
                                       vJ)

    return epsilon0, epsilon1, epsilon2
