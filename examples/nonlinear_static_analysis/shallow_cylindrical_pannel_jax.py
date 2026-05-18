import matplotlib.pyplot as plt
import sympy as sym
import numpy as np

import jax
import jax.numpy as jnp

from shellpy.materials.laminate_orthotropic_material import Lamina, LaminateOrthotropicMaterial

# Garantir precisão dupla no JAX
jax.config.update("jax_enable_x64", True)

from shellpy.continuationpy.continuation2 import Continuation2
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
    u_dof = jnp.array(x[0:-1])
    p = x[-1]
    F_int_jax = jax_func(u_dof)
    F_int_np = np.array(F_int_jax)
    return F_int_np + F_ext * p


def jacobian1(x, jax_func, F_ext):
    u_dof = jnp.array(x[0:-1])
    _, K_tan_jax = jax_func(u_dof)
    J_int = np.array(K_tan_jax)
    return np.hstack((J_int, F_ext[:, np.newaxis]))


def output_results(shell, xi1, xi2, x, *args):
    """
    Nota: Como normalizamos E para GPa (dividimos por 1e9),
    o 'u' retornado pelo solver é 1e9 vezes maior que o real.
    """
    u = x[:-1]
    p = x[-1]

    # Para obter o deslocamento físico real em metros:
    u_fisico = u

    U = shell.displacement_expansion(u_fisico, xi1, xi2)
    N1, N2, N3 = shell.mid_surface_geometry.reciprocal_base(xi1, xi2)
    U_vec = U[0] * N1 + U[1] * N2 + U[2] * N3

    # =====================================================================
    # MODIFICAÇÃO 1: Salvar os dados no TXT ponto a ponto (modo Append)
    # =====================================================================
    u_z = float(-U_vec[2])
    p_val = float(p)
    with open("resultados_curva_bifurcacao.txt", "a") as f:
        f.write(f"{u_z:.8e}\t{p_val:.8e}\n")
    # =====================================================================

    # Plotamos com o 'u' do solver para que a deformação seja visível (exagerada)
    plot_shell_arc(shell, u_fisico * 10)  # Exagero de 10x para visualização

    # Retorno: deslocamento vertical normalizado por polegadas (conforme original)
    return -U_vec[2], p, "u_z", "P"


def plot_shell_arc(shell, u):
    xi1 = np.linspace(*shell.mid_surface_domain.edges["xi1"], 50)
    xi2 = np.linspace(*shell.mid_surface_domain.edges["xi2"], 50)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)
    mode1 = shell.displacement_expansion(u, x, y)
    mode1 = mode1[0:3]
    mode = reciprocal_base[0] * mode1[0] + reciprocal_base[1] * mode1[1] + reciprocal_base[2] * mode1[2]
    z = shell.mid_surface_geometry(x, y)

    fig = plt.figure(1)
    if len(fig.axes) < 2:
        fig.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    ax = plt.subplot(1, 2, 2)
    ax.cla()
    scmap = plt.cm.ScalarMappable(cmap='jet')
    ax.plot_surface(z[0, 0] + mode[0], z[1, 0] + mode[1], z[2, 0] + mode[2],
                    facecolors=scmap.to_rgba(mode[2]),
                    edgecolor='black', linewidth=0.1)
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    plt.pause(0.01)


if __name__ == "__main__":
    # --- Configurações de Integração ---
    integral_x, integral_y, integral_z = 10, 10, 8

    R = 12.0
    L = 5.5
    beta = 0.5
    h = 0.04
    t_ply = 1 / 4

    E11 = 20.46e6
    E22 = 4.092e6
    nu12 = 0.313
    G12 = 2.53704e6
    G13 = 2.53704e6
    G23 = 1.26852e6
    density = 1.0

    # Use physical load units directly (lb). Continuation parameter p is then load in lb.
    load = ConcentratedForce(0.0, 0.0, -1.0 / 4, 0, 0.0)

    rectangular_domain = RectangularMidSurfaceDomain(0, 1, 0, beta)

    # --- Expansão e Condições de Contorno ---
    n_modos, n_modos1 = 9, 9
    expansion_size = {f"u{i}": (n_modos1, n_modos) for i in range(1, 4)}
    expansion_size.update({f"v{i}": (n_modos1, n_modos) for i in range(1, 4)})

    boundary_conditions = {
        "u1": {"xi1": ("S", "F"), "xi2": ("F", "S")},
        "u2": {"xi1": ("F", "F"), "xi2": ("S", "S")},
        "u3": {"xi1": ("FC", "F"), "xi2": ("FC", "C")},
        "v1": {"xi1": ("S", "F"), "xi2": ("F", "S")},
        "v2": {"xi1": ("F", "F"), "xi2": ("S", "S")},
        "v3": {"xi1": ("F", "F"), "xi2": ("F", "S")}
    }

    displacement_field = EnrichedCosineExpansion(expansion_size, rectangular_domain, boundary_conditions)
    eas_field = EasExpansion({"eas": (n_modos1, n_modos)}, rectangular_domain,
                             {"eas": {"xi1": ("F", "F"), "xi2": ("F", "F")}})

    # --- Geometria da Casca ---
    R_sym = sym.Matrix([xi1_ * L, R * sym.sin(xi2_), R * sym.cos(xi2_)])
    mid_surface_geometry = MidSurfaceGeometry(R_sym)
    thickness = ConstantThickness(h)

    ply_0 = Lamina(E_11=E11, E_22=E22, E_33=E22,
                   nu_12=nu12, nu_13=nu12, nu_23=nu12,
                   G_12=G12, G_13=G13, G_23=G23,
                   density=density, angle=0.0, thickness=t_ply)
    ply_90 = Lamina(E_11=E11, E_22=E22, E_33=E22,
                    nu_12=nu12, nu_13=nu12, nu_23=nu12,
                    G_12=G12, G_13=G13, G_23=G23,
                    density=density, angle=np.pi / 2.0, thickness=t_ply)

    material = LaminateOrthotropicMaterial([ply_0, ply_90, ply_0, ply_90, ply_90, ply_0, ply_90, ply_0],
                                           ConstantThickness(h))

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
    print("Modelo JAX compilado com sucesso! Iniciando continuação...")

    residue = lambda u, *args: residuo1(u, jax_func, F_ext)
    jacobian = lambda u, *args: jacobian1(u, jax_func_mat, F_ext)
    stability = shell_stability
    output = lambda u, *args: output_results(shell, 0, 0, u, *args)

    continuation_boundary = np.zeros((n + p, 2))
    continuation_boundary[:-1, 0] = -100000
    continuation_boundary[:-1, 1] = 100000
    continuation_boundary[-1, 0] = -0.1
    continuation_boundary[-1, 1] = 300

    continuation_model = {'n': n,
                          'p': 1,
                          'residue': residue,
                          'jacobian': jacobian,
                          'stability_check': stability,
                          'boundary': continuation_boundary,
                          'output_function': output}

    continuation = Continuation2(continuation_model)

    # CORREÇÃO DOS PARÂMETROS NUMÉRICOS MANTIDA:
    continuation.parameters['tol1'] = 1e-5
    continuation.parameters['tol2'] = 1e-6
    continuation.parameters['index1'] = 0
    continuation.parameters['index2'] = -1
    continuation.parameters['cont_max'] = 10000
    continuation.parameters['plot_real_time'] = True

    continuation.parameters['i_control'] = 5

    continuation.parameters['h0'] = 0.01
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
    with open("resultados_curva_bifurcacao.txt", "w") as f:
        f.write("Deslocamento_uz\tCarga_P\n")
    print("-------- Arquivo TXT inicializado. Salvando passo a passo... --------")
    # =====================================================================

    continuation.continue_branch(u0, t0, w0, 'Branch1')

    plt.show()