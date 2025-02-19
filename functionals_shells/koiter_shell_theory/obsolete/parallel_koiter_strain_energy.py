from itertools import permutations
from joblib import Parallel, delayed
import numpy as np
from functionals_shells.double_integral_booles_rule import double_integral_booles_rule, n_integral_default
from functionals_shells.koiter_shell_theory.koiter_shell_strain_energy import quadratic_energy_density, \
    cubic_energy_density, quartic_energy_density
from functionals_shells.multiindex import MultiIndex
from functionals_shells.shell import Shell


def parallel_quadratic_energy_density(shell, i, j):
    """Função auxiliar para rodar em paralelo"""
    func = lambda xi2, xi1: quadratic_energy_density(shell, i, j, xi1, xi2)
    aux = double_integral_booles_rule(func, shell.boundary, n_integral_default)
    print(i, j, aux)
    return i, j, aux  # Retorna os índices e o valor computado


def parallel_cubic_energy_density(shell, i, j, k):
    """Função auxiliar para rodar em paralelo"""
    func = lambda xi2, xi1: cubic_energy_density(shell, i, j, k, xi1, xi2)
    aux = double_integral_booles_rule(func, shell.boundary, n_integral_default)
    print(i, j, k, aux)
    return i, j, k, aux  # Retorna os índices e o valor computado


def parallel_quartic_energy_density(shell, i, j, k, l):
    """Função auxiliar para rodar em paralelo"""
    func = lambda xi2, xi1: quartic_energy_density(shell, i, j, k, l, xi1, xi2)
    aux = double_integral_booles_rule(func, shell.boundary, n_integral_default)
    print(i, j, k, l, aux)
    return i, j, k, l, aux  # Retorna os índices e o valor computado


def parallel_quadratic_strain_energy(shell: Shell, energy_functional: MultiIndex = None) -> MultiIndex:
    """Computa a energia de deformação quadrática usando processamento paralelo."""
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

    result = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(parallel_quadratic_energy_density)(shell, i, j) for n, (i, j) in enumerate(Vars1))
    print("Fim")
    # Processando os resultados
    for n, (i, j, aux) in enumerate(result):
        exponents = np.zeros(n_dof, dtype=int)
        exponents[i] += 1
        exponents[j] += 1
        energy_functional.add_monomial(exponents, Mult[n] * aux)

    return energy_functional


def parallel_cubic_strain_energy(shell: Shell, energy_functional: MultiIndex = None) -> MultiIndex:
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    boundary = shell.boundary
    if energy_functional is None:
        energy_functional = MultiIndex(n_dof)

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

    result = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(parallel_cubic_energy_density)(shell, i, j, k) for n, (i, j, k) in enumerate(Vars1))
    print("Fim")
    # Processando os resultados
    for n, (i, j, k, aux) in enumerate(result):
        exponents = np.zeros(n_dof, dtype=int)
        exponents[i] += 1
        exponents[j] += 1
        exponents[k] += 1
        energy_functional.add_monomial(exponents, Mult[n] * aux)

    return energy_functional


def parallel_quartic_strain_energy(shell: Shell, energy_functional: MultiIndex = None) -> MultiIndex:
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    boundary = shell.boundary
    if energy_functional is None:
        energy_functional = MultiIndex(n_dof)

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

    result = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(parallel_quartic_energy_density)(shell, i, j, k, l) for n, (i, j, k, l) in enumerate(Vars1))
    print("Fim")
    # Processando os resultados
    for n, (i, j, k, l, aux) in enumerate(result):
        exponents = np.zeros(n_dof, dtype=int)
        exponents[i] += 1
        exponents[j] += 1
        exponents[k] += 1
        exponents[l] += 1
        energy_functional.add_monomial(exponents, Mult[n] * aux)

    return energy_functional
