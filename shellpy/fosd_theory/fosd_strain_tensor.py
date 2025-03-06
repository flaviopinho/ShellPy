import numpy as np

from shellpy import MidSurfaceGeometry, DisplacementExpansion, displacement_covariant_derivatives


def fosd_linear_strain_components(mid_surface_geometry: MidSurfaceGeometry,
                                  displacement_expansion: DisplacementExpansion,
                                  i: int, xi1, xi2):
    curvature_mixed = mid_surface_geometry.curvature_tensor_mixed_components(xi1, xi2)

    # u_i: The displacement associated with the shape function i at coordinates (xi1, xi2)
    u, v = displacement_expansion.shape_function(i, xi1, xi2)

    # u_{i,alpha}: The first derivatives of the displacement u with respect to the curvilinear coordinates
    du, dv = displacement_expansion.shape_function_first_derivatives(i, xi1, xi2)

    # dcu: displacement covariant derivatives
    # ddcu: second covariant derivatives of displacement
    dcu, _, dcv, _ = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, i, xi1, xi2)

    epsilon0 = np.zeros((3, 3) + np.shape(xi1))
    epsilon1 = np.zeros((3, 3) + np.shape(xi1))
    epsilon2 = np.zeros((3, 3) + np.shape(xi1))

    shape_aux = list(range(dcu.ndim))
    shape_aux[0] = 1  # Swap the first dimension
    shape_aux[1] = 0  # Swap the second dimension

    epsilon0[0:2, 0:2] = 1 / 2 * (dcu[0:2] + np.transpose(dcu[0:2], tuple(shape_aux)))

    epsilon1[0:2, 0:2] = 1 / 2 * (dcv[0:2] + np.transpose(dcv[0:2], tuple(shape_aux)))
    upsilon_dcuA = np.einsum('oa..., ob...->ab...', curvature_mixed, dcu[0:2])
    upsilon_dcuB = np.einsum('ob..., oa...->ab...', curvature_mixed, dcu[0:2])
    epsilon1[0:2, 0:2] += 1 / 2 * (upsilon_dcuA + upsilon_dcuB)

    upsilon_dcvA = np.einsum('oa..., ob...->ab...', curvature_mixed, dcv[0:2])
    upsilon_dcvB = np.einsum('ob..., oa...->ab...', curvature_mixed, dcv[0:2])
    epsilon2[0:2, 0:2] = 1 / 2 * (upsilon_dcvA + upsilon_dcvB)

    epsilon0[0:2, 2] = 1 / 2 * (dcu[2, 0:2] + v[0:2])
    epsilon0[2, 0:2] = epsilon0[0:2, 2]

    epsilon1[0:2, 2] = 1 / 2 * dv[2, 0:2]
    epsilon1[2, 0:2] = epsilon1[0:2, 2]

    epsilon0[2, 2] = v[2]

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

    epsilon0[2, 0:2] = 1 / 2 * np.einsum('ot...,t...,ob...->b...',
                                         metric_tensor_contravariant_components,
                                         vI, dcuJ)

    epsilon0[0:2, 2] = 1 / 2 * np.einsum('ot...,ob...,t...->b...',
                                         metric_tensor_contravariant_components,
                                         dcuI, vJ)

    epsilon1[2, 0:2] = 1 / 2 * np.einsum('ot...,t...,ob...->b...',
                                         metric_tensor_contravariant_components,
                                         vI, dcvJ)

    epsilon1[0:2, 2] = 1 / 2 * np.einsum('ot...,ob...,t...->b...',
                                         metric_tensor_contravariant_components,
                                         dcvI, vJ)

    epsilon0[2, 2] = 1 / 2 * np.einsum('ot...,o...,t...->...',
                                       metric_tensor_contravariant_components,
                                       vI, vJ)

    return epsilon0, epsilon1, epsilon2
