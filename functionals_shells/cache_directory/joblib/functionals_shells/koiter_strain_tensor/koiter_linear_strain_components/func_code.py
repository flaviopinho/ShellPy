# first line: 14
@memory.cache
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

    # rho_{alpha beta} = C^gamma_{alpha beta} u_{3| gamma} + u_{3| alpha beta}
    rho = np.einsum('gab...,g...->ab...', C, dcu3) - ddcu3

    return gamma, rho
