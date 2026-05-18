import matplotlib.pyplot as plt
import sympy as sym
import numpy as np

import jax
import jax.numpy as jnp

from shellpy.continuationpy import Continuation2

# Garantir precisão dupla no JAX
jax.config.update("jax_enable_x64", True)

from shellpy.continuationpy.continuation import Continuation
from shellpy.expansions import EnrichedCosineExpansion
from shellpy.fsdt6 import load_energy
from shellpy.fsdt7_eas.EAS_expansion import EasExpansion
# --- Imports Substituídos pelo JAX ---
from shellpy.fsdt7_eas_jax.jax_strain_energy import fsdt7_strain_energy_internal_force_tangent_matrix_jax
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
# -------------------------------------
from shellpy.utils.residue_jacobian_stability import shell_stability
from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.tensor_derivatives import tensor_derivative
from shellpy.shell_loads.shell_conservative_load import ConcentratedForce
from shellpy import Shell
from shellpy import ConstantThickness
from shellpy import MidSurfaceGeometry, xi1_, xi2_


# =========================================================================
# WRAPPERS PARA O JAX
# =========================================================================
def residuo1(x, jax_func, F_ext):
    """
    Calcula o vetor resíduo usando a função JAX compilada.
    x[:-1] = deslocamentos (u)
    x[-1]  = fator de carga (p)
    """
    u_dof = jnp.array(x[0:-1])
    p = x[-1]

    # O JAX retorna F_int e K_tan simultaneamente. Pegamos apenas F_int aqui.
    F_int_jax = jax_func(u_dof)

    # Converte de volta para NumPy para não quebrar a biblioteca Continuation
    F_int_np = np.array(F_int_jax)

    return F_int_np + F_ext * p


def jacobian1(x, jax_func_mat, F_ext):
    """
    Calcula a matriz Jacobiana usando a função JAX compilada.
    """
    u_dof = jnp.array(x[0:-1])

    # Pegamos apenas K_tan (a condensação do EAS já acontece internamente no JAX)
    _, K_tan_jax = jax_func_mat(u_dof)
    J_int = np.array(K_tan_jax)

    return np.hstack((J_int, F_ext[:, np.newaxis]))


def plot_shell_arc(shell, u):
    """
    Plots the deformed shell geometry.
    """
    xi1 = np.linspace(*shell.mid_surface_domain.edges["xi1"], 50)
    xi2 = np.linspace(*shell.mid_surface_domain.edges["xi2"], 50)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)

    mode1 = shell.displacement_expansion(u, x, y)
    mode1 = mode1[0:3]
    mode = reciprocal_base[0] * mode1[0] + reciprocal_base[1] * mode1[1] + reciprocal_base[2] * mode1[2]

    z = shell.mid_surface_geometry(x, y)

    fig = plt.figure(1)
    n = len(fig.axes)
    if n == 1 or n == 0:
        fig.clf()
        ax = fig.add_subplot(1, 2, 1)
        ax = fig.add_subplot(1, 2, 2, projection='3d')

        try:
            data = np.loadtxt("semicylindrical_ansys.txt", delimiter=";")
            x_ansys = data[:, 0]
            y_ansys = data[:, 1] / 1000
            ax = plt.subplot(1, 2, 1)
            ax.plot(x_ansys, y_ansys, linestyle='-', color='k', label='ANSYS')
            ax.legend()
        except FileNotFoundError:
            pass  # Ignora caso o arquivo do ansys não esteja na pasta

        ax = plt.subplot(1, 2, 2)

    else:
        ax = plt.subplot(1, 2, 2)

    ax.cla()

    scmap = plt.cm.ScalarMappable(cmap='jet')

    ax.plot_surface(z[0, 0] + mode[0], z[1, 0] + mode[1], z[2, 0] + mode[2],
                    facecolors=scmap.to_rgba(mode[2]),
                    edgecolor='black',
                    linewidth=0.1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    plt.pause(0.01)


def output_results(shell, xi1, xi2, x, *args):
    u = x[:-1]
    p = x[-1]
    U = shell.displacement_expansion(u, xi1, xi2)
    N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(xi1, xi2)
    U = U[0] * N1 + U[1] * N2 + U[2] * N3

    # Plotagem da casca com validação do ANSYS restaurada
    plot_shell_arc(shell, u)

    print(f"  u={-U[2]}, F={p}")

    # =====================================================================
    # MODIFICAÇÃO 1: Salvar os dados no TXT ponto a ponto (modo Append)
    # =====================================================================
    u_z = float(-U[2])
    p_val = float(p)
    with open("resultados_curva_bifurcacao3.txt", "a") as f:
        f.write(f"{u_z:.8e}\t{p_val:.8e}\n")
    # =====================================================================

    return -U[2], p, "u_z", "P"


if __name__ == "__main__":
    integral_x = 10
    integral_y = 10
    integral_z = 4

    R = 1.016
    L = 3.048
    h = 0.03

    density = 1

    # NORMALIZAÇÃO DE VARIÁVEIS MANTIDA:
    # Usando a escala física real do material para que o Resíduo de forças
    # não seja mascarado por tolerâncias frouxas.
    E = 2.0685E7
    nu = 0.3

    # A carga de -500 agora atua diretamente na escala de E.
    load = ConcentratedForce(0, 0, -500.0, 1, 0)

    rectangular_domain = RectangularMidSurfaceDomain(0, 1, 0, np.pi / 2)

    n_modos = 10

    expansion_size = {"u1": (n_modos, n_modos),
                      "u2": (n_modos, n_modos),
                      "u3": (n_modos, n_modos),
                      "v1": (n_modos, n_modos),
                      "v2": (n_modos, n_modos),
                      "v3": (n_modos, n_modos)}

    boundary_conditions_u1 = {"xi1": ("S", "F"), "xi2": ("F", "F")}
    boundary_conditions_u2 = {"xi1": ("S", "F"), "xi2": ("S", "S")}
    boundary_conditions_u3 = {"xi1": ("C", "F"), "xi2": ("FC", "FC")}
    boundary_conditions_v1 = {"xi1": ("S", "F"), "xi2": ("F", "F")}
    boundary_conditions_v2 = {"xi1": ("S", "F"), "xi2": ("S", "S")}
    boundary_conditions_v3 = {"xi1": ("F", "F"), "xi2": ("F", "F")}

    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3,
                           "v1": boundary_conditions_v1,
                           "v2": boundary_conditions_v2,
                           "v3": boundary_conditions_v3}

    displacement_field = EnrichedCosineExpansion(expansion_size, rectangular_domain, boundary_conditions)

    eas_field = EasExpansion({"eas": (n_modos, n_modos)}, rectangular_domain,
                             {"eas": {"xi1": ("F", "F"), "xi2": ("F", "F")}})

    R_ = sym.Matrix([xi1_ * L, R * sym.sin(xi2_), R * sym.cos(xi2_)])
    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, density)
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, load)

    U_ext = load_energy(shell)
    F_ext = tensor_derivative(U_ext, 0)

    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()
    n = displacement_field.number_of_degrees_of_freedom()
    p = 1

    # =========================================================================
    # SETUP DO JAX (Factory & Warmup)
    # =========================================================================
    print("Preparando e compilando o modelo JAX (Isso pode levar alguns segundos)...")
    jax_func, jax_func_mat = fsdt7_strain_energy_internal_force_tangent_matrix_jax(
        shell, eas_field, integral_x, integral_y, integral_z, gauss_weights_simple_integral
    )

    # "Warmup" (Aquecimento) do JAX
    _ = jax_func(jnp.zeros(n_dof))
    _ = jax_func_mat(jnp.zeros(n_dof))
    print("Modelo JAX compilado com sucesso! Iniciando continuação...")

    residue = lambda u, *args: residuo1(u, jax_func, F_ext)
    jacobian = lambda u, *args: jacobian1(u, jax_func_mat, F_ext)
    stability = shell_stability
    output = lambda u, *args: output_results(shell, 1, 0, u, *args)

    continuation_boundary = np.zeros((n + p, 2))
    continuation_boundary[:-1, 0] = -100000
    continuation_boundary[:-1, 1] = 100000
    continuation_boundary[-1, 0] = -0.1
    continuation_boundary[-1, 1] = 1.6

    continuation_model = {'n': n,
                          'p': 1,
                          'residue': residue,
                          'jacobian': jacobian,
                          'stability_check': stability,
                          'boundary': continuation_boundary,
                          'output_function': output}

    continuation = Continuation2(continuation_model)

    # CORREÇÃO DOS PARÂMETROS NUMÉRICOS MANTIDA:
    continuation.parameters['tol1'] = 1e-3
    continuation.parameters['tol2'] = 1e-6
    continuation.parameters['index1'] = 0
    continuation.parameters['index2'] = -1
    continuation.parameters['cont_max'] = 10000
    continuation.parameters['plot_real_time'] = True

    continuation.parameters['i_control'] = 5

    continuation.parameters['h0'] = 1
    continuation.parameters['h_max'] = 1000  # Controle de passo mantido para estabilidade
    continuation.parameters['jacobian_corrector'] = True

    u0 = np.zeros(continuation_model['n'] + continuation_model['p'])
    u0[-1] = 0.01
    H0 = continuation.model['residue'](u0)
    J0 = continuation.model['jacobian'](u0)
    t0 = continuation.tangent_vector(J0)
    w0 = 1

    # =====================================================================
    # MODIFICAÇÃO 2: Criar e limpar o arquivo antes do loop começar
    # =====================================================================
    with open("resultados_curva_bifurcacao3.txt", "w") as f:
        f.write("Deslocamento_uz\tCarga_P\n")
    print("-------- Arquivo TXT inicializado. Salvando passo a passo... --------")
    # =====================================================================

    continuation.continue_branch(u0, t0, w0, 'Branch1')

    plt.show()