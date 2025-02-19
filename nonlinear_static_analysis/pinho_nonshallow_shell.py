import sys
import matplotlib.pyplot as plt
import sympy as sym
import numpy as np

from displacement_expansion import simply_supported, pinned
from expansions.eigen_function_expansion import EigenFunctionExpansion
from expansions.polinomial_expansion import GenericPolynomialSeries
from mid_surface_domain import RectangularMidSurfaceDomain
from koiter_shell_theory import fast_koiter_strain_energy
from tensor_derivatives import tensor_derivative
from koiter_shell_theory.koiter_load_energy import koiter_load_energy
from shell_loads.shell_conservative_load import ConcentratedForce, PressureLoad
from material import LinearElasticMaterial
from shell import Shell
from thickness import ConstantThickness
from midsurface_geometry import MidSurfaceGeometry, xi1_, xi2_

sys.path.append('../../ContinuationPy/ContinuationPy')
import continuation


def pinho_nonshallow_shell_residue(F_int, F_ext, x, *args):
    u = x[:-1]
    p = x[-1]
    F_int_tot = np.einsum('ij, j->i', F_int[0], u, optimize=True) + \
                np.einsum('ijk, j, k->i', F_int[1], u, u, optimize=True) + \
                np.einsum('ijkl, j, k, l->i', F_int[2], u, u, u, optimize=True)
    return F_int_tot + F_ext * p


def pinho_nonshallow_shell_jacobian(J_int, F_ext, x, *args):
    u = x[:-1]
    p = [-1]
    J_int_tot = J_int[0] + \
                np.einsum('ijk, k->ij', J_int[1], u, optimize=True) + \
                np.einsum('ijkl, k, l->ij', J_int[2], u, u, optimize=True)

    return np.hstack((J_int_tot, F_ext[:, np.newaxis]))


def pinho_nonshallow_shell_stability(u, J, model, *args):
    # Extrai a submatriz Jx
    Jx = -J[:model['n'], :model['n']]

    # Determinação dos autovalores de Jx
    eigen_values = np.linalg.eigvals(Jx)

    # Partes reais e imaginárias dos autovalores
    real_part = np.real(eigen_values)
    imaginary_part = np.imag(eigen_values)

    # Estabilidade: maior parte real
    stability = np.max(real_part)

    tipo = None
    if 'tipo' in locals():  # Para verificar se a variável tipo foi definida
        # Análise do tipo de bifurcação
        index_real_positivo = real_part > 0
        index_real_negativo = real_part < 0
        num_real_positivo = np.sum(index_real_positivo)
        num_real_negativo = np.sum(index_real_negativo)

        if num_real_positivo == 2 and np.any(imaginary_part[index_real_positivo] != 0):
            tipo = 'H'  # Hopf
        elif num_real_positivo == 1 and num_real_negativo >= 0:
            tipo = 'SN'  # Ponto de sela
        elif num_real_positivo == 0 and num_real_negativo >= 0:
            tipo = 'PR'  # Ponto regular
        else:
            tipo = 'BC'  # Bifurcação complexa

    return stability, tipo


def pinho_output_results(shell, xi1, xi2, x, *args):
    u = x[:-1]
    p = x[-1]
    U = shell.displacement_expansion(u, xi1, xi2)
    N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(xi1, xi2)
    U = U[0] * N1 + U[1] * N2 + U[2] * N3

    return p, U[2], "p (1E7)", "u_3(a/4,b/4)/h"


if __name__ == "__main__":

    R = 0.1
    a = 0.1
    b = 0.1
    h = 0.001
    density = 7850

    E = 1
    E1 = 206E9
    nu = 0.3

    load = PressureLoad(-1E7 / E1)

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    expansion_size = {"u1": (6, 6),
                      "u2": (6, 6),
                      "u3": (6, 6)}

    boundary_conditions = pinned

    mapping = []
    modes_xi1 = [2, 4, 6, 8, 10, 12, 14, 16]
    modes_xi2 = [1, 3, 5, 7, 9, 11, 13, 15]
    for i in range(expansion_size["u1"][0]):
        for j in range(expansion_size["u1"][1]):
            mapping.append(("u1", modes_xi1[i], modes_xi2[j]))

    modes_xi1 = [1, 3, 5, 7, 9, 11, 13, 15]
    modes_xi2 = [2, 4, 6, 8, 10, 12, 14, 16]
    for i in range(expansion_size["u2"][0]):
        for j in range(expansion_size["u2"][1]):
            mapping.append(("u2", modes_xi1[i], modes_xi2[j]))

    modes_xi1 = [1, 3, 5, 7, 9, 11, 13, 15]
    modes_xi2 = [1, 3, 5, 7, 9, 11, 13, 15]
    for i in range(expansion_size["u3"][0]):
        for j in range(expansion_size["u3"][1]):
            mapping.append(("u3", modes_xi1[i], modes_xi2[j]))

    displacement_field = EigenFunctionExpansion(expansion_size, rectangular_domain, boundary_conditions, mapping)
    # displacement_field = GenericPolynomialSeries(np.polynomial.Legendre, expansion_size, mid_surface_domain, boundary_conditions)

    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = LinearElasticMaterial(E, nu, density)
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, load)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    U_ext = koiter_load_energy(shell)

    U2_int, U3_int, U4_int = fast_koiter_strain_energy(shell)

    # Numero de variaveis
    n = displacement_field.number_of_degrees_of_freedom()
    # Numero de parametros
    p = 1

    div = E * h ** 2
    F_ext = tensor_derivative(U_ext, 0) / div

    F2_int = tensor_derivative(U2_int, 0) * h / div
    F3_int = tensor_derivative(U3_int, 0) * h ** 2 / div
    F4_int = tensor_derivative(U4_int, 0) * h ** 3 / div

    J2_int = tensor_derivative(F2_int, 1)
    J3_int = tensor_derivative(F3_int, 1)
    J4_int = tensor_derivative(F4_int, 1)

    residue = lambda u, *args: pinho_nonshallow_shell_residue((F2_int, F3_int, F4_int), F_ext, u, *args)
    jacobian = lambda u, *args: pinho_nonshallow_shell_jacobian((J2_int, J3_int, J4_int), F_ext, u, *args)
    stability = pinho_nonshallow_shell_stability
    output = lambda u, *args: pinho_output_results(shell, a/4, b/4, u, *args)

    # Limites de interesse das variaveis e parametros
    continuation_boundary = np.zeros((n + p, 2))
    continuation_boundary[:-1, 0] = -1000
    continuation_boundary[:-1, 1] = 1000
    continuation_boundary[-1, 0] = -2
    continuation_boundary[-1, 1] = 2

    # Definindo continuation_model
    continuation_model = {'n': n_dof,
                          'p': 1,
                          'residue': residue,
                          'jacobian': jacobian,
                          'stability_check': stability,
                          'boundary': continuation_boundary,
                          'output_function': output}

    continuation = continuation.Continuation(continuation_model)
    continuation.parameters['tol2'] = 0
    continuation.parameters['tol1'] = 1E-5
    continuation.parameters['index1'] = -1
    continuation.parameters['index2'] = mapping.index(('u3', 1, 1))

    continuation.parameters['h_max'] = 0.1

    # Determinacao de um ponto regular inicial
    u0 = np.zeros(continuation_model['n'] + continuation_model['p'])
    u0[-1] = 0.01
    H0 = continuation.model['residue'](u0)
    J0 = continuation.model['jacobian'](u0)
    t0 = continuation.tangent_vector(J0)
    w0 = 1

    continuation.continue_branch(u0, t0, w0, 'Branch1')

    plt.show()
