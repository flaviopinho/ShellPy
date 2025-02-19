import numpy as np
from functionals_shells.double_integral_booles_rule import double_integral_booles_rule, n_integral_default
from functionals_shells.multiindex import MultiIndex
from functionals_shells.shell import Shell


def koiter_kinetic_energy(shell, energy_functional: MultiIndex = None) -> MultiIndex:
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    boundary = shell.boundary
    if energy_functional is None:
        energy_functional = MultiIndex(n_dof)

    Mult = []
    Vars1 = []
    for i in range(n_dof):
        for j in range(i, n_dof):
            Vars1.append((i, j))
            if i == j:
                Mult.append(1)
            else:
                Mult.append(2)

    for n, (i, j) in enumerate(Vars1):
        func = lambda xi2, xi1: kinetic_energy_density(shell, i, j, xi1, xi2)
        aux = double_integral_booles_rule(func, boundary, n_integral_default)

        exponents = np.zeros(n_dof, dtype=int)
        exponents[i] += 1
        exponents[j] += 1
        print(i, j, aux)
        energy_functional.add_monomial(exponents, Mult[n] * aux)

    return energy_functional


def kinetic_energy_density(shell: Shell, i, j, xi1, xi2):
    rho = shell.material.density
    h = shell.thickness(xi1, xi2)
    G = np.zeros((3, 3) + np.shape(xi1))
    G[0:2, 0:2] = shell.mid_surface_geometry.metric_tensor_contravariant_components(xi1, xi2)
    G[2, 2] = 1
    u_i = shell.displacement_expansion.shape_function(i, xi1, xi2)
    u_j = shell.displacement_expansion.shape_function(j, xi1, xi2)

    # G^{ij} u_i u_j sqrt(G)
    return (rho * h / 2 * np.einsum('ij...,i...,j...->...', G, u_i, u_j)) * shell.mid_surface_geometry.sqrtG(xi1,
                                                                                                                 xi2)
