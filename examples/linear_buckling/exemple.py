from time import time

import numpy as np
import sympy as sym
from matplotlib import pyplot as plt
from scipy.linalg import eigh  # Necessário para a análise de instabilidade

from shellpy import RectangularMidSurfaceDomain, xi1_, xi2_, MidSurfaceGeometry, ConstantThickness, Shell
from shellpy.displacement_expansion import simply_supported
from shellpy.expansions import EigenFunctionExpansion
from shellpy.materials import IsotropicHomogeneousLinearElasticMaterial
from shellpy.sanders_koiter import koiter_load_energy, fast_koiter_quadratic_strain_energy
from shellpy.sanders_koiter.koiter_geometric_stiffness_matrix import koiter_geometric_stiffness_matrix
from shellpy.shell_loads import ConcentratedForceGlobal
from shellpy.tensor_derivatives import tensor_derivative

if __name__ == "__main__":


    n_integral_default = 20
    plot_scale = 10  # Ajustado para visualização do modo de flambagem

    # Entrada de dados - PLACA PLANA
    a = 1.0
    b = 1.0
    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)
    R_ = sym.Matrix([xi1_, xi2_, 0])

    h = 0.01  # 10 mm
    density = 7850
    E = 200E9  # Aço
    nu = 0.3

    # CARGA NO PLANO (X): Essa é a chave!
    # Aplicamos uma força de -1000 N no eixo X, no meio da placa.
    # Isso vai comprimir a região de x=0 até x=a/2, gerando o estado N0.
    load = ConcentratedForceGlobal(-1000, 0, 0, a / 2, b / 2)

    # 6x6 termos já dão uma boa convergência para placas
    expansion_size = {"u1": (6, 6), "u2": (6, 6), "u3": (6, 6)}
    boundary_conditions = simply_supported

    displacement_field = EigenFunctionExpansion(expansion_size, rectangular_domain, boundary_conditions)

    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, density)
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, load)

    # =================================================================
    # 1. ANÁLISE ESTÁTICA LINEAR (ESTADO FUNDAMENTAL)
    # =================================================================
    n_dof = shell.displacement_expansion.number_of_degrees_of_freedom()

    U1 = koiter_load_energy(shell)

    print("Calculando Matriz de Rigidez Elástica (K_E)...")
    start = time()
    U2 = fast_koiter_quadratic_strain_energy(shell)
    stop = time()
    print("Tempo K_E: ", stop - start)

    F = -tensor_derivative(U1, 0)
    K_E = tensor_derivative(tensor_derivative(U2, 0), 1)

    # Vetor de deslocamentos pré-flambagem (Este é o U0 que sua função precisa!)
    U0 = np.linalg.inv(K_E) @ F

    # =================================================================
    # 2. ANÁLISE DE FLAMBAGEM (USANDO A SUA FUNÇÃO GENÉRICA)
    # =================================================================
    # A sua função vai ler U0, calcular N0 real ponto a ponto e montar K_G
    K_G = koiter_geometric_stiffness_matrix(shell, U0)

    # Resolução do problema de autovalor invertido para evitar erro de Cholesky
    # Resolvemos: K_G * v = gamma * K_E * v (onde gamma = 1 / lambda_cr)
    print("Extraindo autovalores...")
    gammas, eigenvectors = eigh(K_G, K_E)

    # Filtramos para pegar a flambagem sob compressão (gammas positivos)
    positive_gammas = gammas[gammas > 1e-12]

    if len(positive_gammas) == 0:
        raise ValueError("Nenhum modo de flambagem encontrado. Verifique as condições de contorno.")

    # O MENOR fator de carga lambda corresponde ao MAIOR gamma
    gamma_max = np.max(positive_gammas)
    lambda_cr = 1.0 / gamma_max
    index_cr = np.where(gammas == gamma_max)[0][0]

    # Este é o vetor de coeficientes do modo crítico!
    mode_cr = eigenvectors[:, index_cr]

    print("\n" + "=" * 50)
    print("ANÁLISE DE INSTABILIDADE CONCLUÍDA")
    print("=" * 50)
    print(f"Fator de Carga Crítica (lambda_cr) : {lambda_cr:.2f}")
    print(f"Carga Crítica de Flambagem         : {lambda_cr * 1000:.2f} N")
    print("=" * 50 + "\n")

    # =================================================================
    # 3. GERAÇÃO DA MALHA E PLOTAGEM DO MODO CRÍTICO
    # =================================================================
    xi1 = np.linspace(*rectangular_domain.edges["xi1"], 50)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], 50)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Passamos o autovetor (mode_cr) para desenhar a forma flambada, e NÃO o U0 linear
    mode1_cr = shell.displacement_expansion(mode_cr, x, y)
    mode = reciprocal_base[0] * mode1_cr[0] + reciprocal_base[1] * mode1_cr[1] + reciprocal_base[2] * mode1_cr[2]

    # Normalizar o modo para fins puramente de visualização
    mode_norm = mode / np.max(np.abs(mode)) * h * plot_scale
    z = shell.mid_surface_geometry(x, y)

    cmap = plt.cm.jet
    norm = colors.Normalize(vmin=np.min(mode1_cr[2]), vmax=np.max(mode1_cr[2]))
    facecolors = cmap(norm(mode1_cr[2]))

    surf = ax.plot_surface(
        z[0, 0] + mode_norm[0],
        z[1, 0] + mode_norm[1],
        z[2, 0] + mode_norm[2],
        facecolors=facecolors,
        edgecolor='k', linewidth=0.2, alpha=0.9
    )

    scmap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scmap.set_array(mode1_cr[2])
    cbar = fig.colorbar(scmap, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label("Deslocamento Normal Normalizado (Modo de Flambagem)")

    ax.set_title("1º Modo de Flambagem via koiter_geometric_stiffness_matrix")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(x) * 0.3))

    plt.show()