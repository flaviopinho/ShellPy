import numpy as np

from functionals_shells.cache_decorator import cache_function
from functionals_shells.displacement_covariant_derivative import displacement_covariant_derivatives
from functionals_shells.displacement_expansion import DisplacementExpansion
from functionals_shells.midsurface_geometry import MidSurfaceGeometry


@cache_function
def koiter_linear_strain_components(mid_surface_geometry: MidSurfaceGeometry,
                                    displacement_expansion: DisplacementExpansion,
                                    i: int, xi1, xi2):
    # u_{i|alpha}
    # u_{i | alpha beta}
    dcu, ddcu = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, i, xi1, xi2)

    # u_{3|alpha}
    dcu3 = dcu[2]

    # u_{beta|alpha}
    dcu = dcu[0:2]

    # u_{3|alpha beta}
    ddcu3 = ddcu[2]

    # u_{tau|alpha beta}
    ddcu = ddcu[0:2]

    shape_aux = list(range(dcu.ndim))
    shape_aux[0] = 1
    shape_aux[1] = 0

    # gamma_{alpha beta} = 1/2 (u_{alpha|beta} + u_{beta|alpha})
    gamma = 1 / 2 * (dcu + np.transpose(dcu, tuple(shape_aux)))

    # C^i_{j alpha}
    C = mid_surface_geometry.christoffel_symbols(xi1, xi2)

    # C^gamma_{beta alpha}
    C = C[0:2, 0:2]

    # rho_{alpha beta} = C^gamma_{alpha beta} u_{3| gamma} - u_{3| alpha beta}
    rho = np.einsum('gab...,g...->ab...', C, dcu3) - ddcu3

    return gamma, rho


@cache_function
def koiter_nonlinear_strain_components_total(mid_surface_geometry: MidSurfaceGeometry,
                                             displacement_expansion: DisplacementExpansion,
                                             i: int, j: int, xi1, xi2):
    # ui_{i|alpha}
    # ui_{i | alpha beta}
    dcu1, ddcu1 = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, i, xi1, xi2)

    # uj_{i|alpha}
    # uj_{i | alpha beta}
    dcu2, ddcu2 = displacement_covariant_derivatives(mid_surface_geometry, displacement_expansion, j, xi1, xi2)

    # G^{alpha beta}
    metric_tensor_contravariant_components = mid_surface_geometry.metric_tensor_contravariant_components(xi1, xi2)
    # G^{i p}
    shape = list(np.shape(metric_tensor_contravariant_components))
    shape[0] = 3
    shape[1] = 3
    metric_tensor_contravariant_components2 = np.zeros(shape)
    metric_tensor_contravariant_components2[0:2, 0:2] = metric_tensor_contravariant_components
    metric_tensor_contravariant_components2[2, 2] = 1

    # u1^{p}_{|alpha} = G^{p i} u1_{i|alpha}
    # u1^{p}_{|alpha} u2_{p|beta}
    gamma_nl = 0.5 * np.einsum('pi...,ia...,pb...->ab...', metric_tensor_contravariant_components2, dcu1, dcu2)

    return gamma_nl
