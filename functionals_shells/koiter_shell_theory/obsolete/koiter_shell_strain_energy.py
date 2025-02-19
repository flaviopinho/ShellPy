from itertools import permutations
import numpy as np
from functionals_shells.double_integral_booles_rule import double_integral_booles_rule, n_integral_default
from functionals_shells.koiter_shell_theory import koiter_linear_strain_components, \
    koiter_nonlinear_strain_components_total
from functionals_shells.multiindex import MultiIndex
from functionals_shells.shell import Shell


def strain_energy(shell: Shell) -> MultiIndex:
    a = quadratic_strain_energy(shell)
    a = cubic_strain_energy(shell, a)
    a = quartic_strain_energy(shell, a)
    return a


def quadratic_strain_energy(shell: Shell, energy_functional: MultiIndex = None) -> MultiIndex:
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
        func = lambda xi2, xi1: quadratic_energy_density(shell, i, j, xi1, xi2)
        aux = double_integral_booles_rule(func, boundary, n_integral_default)

        exponents = np.zeros(n_dof, dtype=int)
        exponents[i] += 1
        exponents[j] += 1
        print(i, j, Mult[n] * aux)
        energy_functional.add_monomial(exponents, Mult[n] * aux)

    return energy_functional


def cubic_strain_energy(shell: Shell, energy_functional: MultiIndex = None) -> MultiIndex:
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    boundary = shell.boundary
    if energy_functional is None:
        energy_functional = MultiIndex(n_dof)

    #nonlinear_dofs = []
    #for n in range(n_dof):
    #    if shell.displacement_expansion.mapping(n)[0] == "u3":
    #        nonlinear_dofs.append(n)

    nonlinear_dofs = range(n_dof)

    Mult = []
    Vars1 = []
    for i in range(n_dof):
        for j in nonlinear_dofs:
            for k in nonlinear_dofs:
                if k < j:
                    continue
                Vars1.append((i, j, k))
                if j == k:
                    Mult.append(1)
                else:
                    Mult.append(2)

    for n, (i, j, k) in enumerate(Vars1):
        func = lambda xi2, xi1: cubic_energy_density(shell, i, j, k, xi1, xi2)
        aux = double_integral_booles_rule(func, boundary, n_integral_default)
        exponents = np.zeros(n_dof, dtype=int)
        exponents[i] += 1
        exponents[j] += 1
        exponents[k] += 1
        print(i, j, k, Mult[n] * aux)
        energy_functional.add_monomial(exponents, Mult[n] * aux)

    return energy_functional


def quartic_strain_energy(shell: Shell, energy_functional: MultiIndex = None) -> MultiIndex:
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    boundary = shell.boundary
    if energy_functional is None:
        energy_functional = MultiIndex(n_dof)

    #nonlinear_dofs = []
    #for n in range(n_dof):
    #    if shell.displacement_expansion.mapping(n)[0] == "u3":
    #        nonlinear_dofs.append(n)

    nonlinear_dofs = range(n_dof)

    Mult = []
    Vars1 = []
    for i in nonlinear_dofs:
        for j in nonlinear_dofs:
            if j < i:
                continue
            for k in nonlinear_dofs:
                if k < j:
                    continue
                for l in nonlinear_dofs:
                    if l < k:
                        continue
                    Vars1.append((i, j, k, l))
                    unique_perms = np.unique(list(permutations([i, j, k, l])), axis=0)
                    Mult.append(len(unique_perms))

    for n, (i, j, k, l) in enumerate(Vars1):
        func = lambda xi2, xi1: quartic_energy_density(shell, i, j, k, l, xi1, xi2)
        aux = double_integral_booles_rule(func, boundary, n_integral_default)
        exponents = np.zeros(n_dof, dtype=int)
        exponents[i] += 1
        exponents[j] += 1
        exponents[k] += 1
        exponents[l] += 1
        print(i, j, k, l, Mult[n] * aux)
        energy_functional.add_monomial(exponents, Mult[n] * aux)

    return energy_functional


def quadratic_energy_density(shell: Shell, i: int, j: int, xi1, xi2):
    h = shell.thickness(xi1, xi2)
    g = shell.mid_surface_geometry.metric_tensor_contravariant_components(xi1, xi2)
    constitutive_tensor = shell.material.thin_shell_constitutive_tensor(g)
    rho1, kappa1 = koiter_linear_strain_components(shell.mid_surface_geometry, shell.displacement_expansion, i, xi1,
                                                   xi2)
    rho2, kappa2 = koiter_linear_strain_components(shell.mid_surface_geometry, shell.displacement_expansion, j, xi1,
                                                   xi2)

    aux1 = np.einsum('ij...,kl...->ijkl...', rho1, rho2)
    aux2 = np.einsum('ij...,kl...->ijkl...', kappa1, kappa2)

    auxiliar_tensor = h / 2 * aux1 + h ** 3 / 24 * aux2

    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)
    den=(np.einsum('ijkl..., ijkl...->...', constitutive_tensor, auxiliar_tensor))
    return den*sqrtG


def cubic_energy_density(shell: Shell, i: int, j: int, k: int, xi1, xi2):
    h = shell.thickness(xi1, xi2)
    g = shell.mid_surface_geometry.metric_tensor_contravariant_components(xi1, xi2)
    constitutive_tensor = shell.material.thin_shell_constitutive_tensor(g)
    rho1, kappa1 = koiter_linear_strain_components(shell.mid_surface_geometry, shell.displacement_expansion, i, xi1,
                                                   xi2)
    rho2 = koiter_nonlinear_strain_components_total(shell.mid_surface_geometry, shell.displacement_expansion, j, k, xi1, xi2)

    aux1 = np.einsum('ijkl..., ij...,kl...->...', constitutive_tensor, rho1, rho2)

    return 2 * h / 2 * aux1 * shell.mid_surface_geometry.sqrtG(xi1, xi2)


def quartic_energy_density(shell: Shell, i: int, j: int, k: int, l: int, xi1, xi2):
    h = shell.thickness(xi1, xi2)
    g = shell.mid_surface_geometry.metric_tensor_contravariant_components(xi1, xi2)
    constitutive_tensor = shell.material.thin_shell_constitutive_tensor(g)
    rho1 = koiter_nonlinear_strain_components_total(shell.mid_surface_geometry, shell.displacement_expansion, i, j, xi1, xi2)
    rho2 = koiter_nonlinear_strain_components_total(shell.mid_surface_geometry, shell.displacement_expansion, k, l, xi1, xi2)

    aux1 = np.einsum('ijkl...,ij...,kl...->...', constitutive_tensor, rho1, rho2)

    return h / 2 * aux1 * shell.mid_surface_geometry.sqrtG(xi1, xi2)
