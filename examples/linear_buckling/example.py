from time import time
import numpy as np
import sympy as sym
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from matplotlib import colors

from shellpy import RectangularMidSurfaceDomain, xi1_, xi2_, MidSurfaceGeometry, ConstantThickness, Shell
from shellpy.displacement_expansion import simply_supported
from shellpy.expansions import EigenFunctionExpansion
from shellpy.materials import IsotropicHomogeneousLinearElasticMaterial
from shellpy.sanders_koiter import koiter_load_energy, fast_koiter_quadratic_strain_energy
from shellpy.sanders_koiter.koiter_geometric_stiffness_matrix import koiter_geometric_stiffness_matrix
from shellpy.tensor_derivatives import tensor_derivative
from shellpy.shell_loads import LineLoadGlobal, LoadCollection

if __name__ == "__main__":
    plot_scale = 10

    # =================================================================
    # 1. DEFINIÇÃO DO MODELO (PLACA QUADRADA)
    # =================================================================
    a, b = 1.0, 1.0
    h = 0.01  # 10 mm
    E = 200E9
    nu = 0.3

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)
    R_ = sym.Matrix([xi1_, xi2_, 0])

    # Carga de compressão unitária (-1 N/m) aplicada nas duas bordas opostas
    load_right = LineLoadGlobal(qx=-1.0, qy=0.0, qz=0.0, line_along='xi2', constant_coord=a, start_coord=0.0,
                                end_coord=b)
    load_left = LineLoadGlobal(qx=1.0, qy=0.0, qz=0.0, line_along='xi2', constant_coord=0.0, start_coord=0.0,
                               end_coord=b)
    load = LoadCollection([load_left, load_right])

    # Condições de contorno: Membrana livre para deformar, Flexão apoiada
    bc_u1 = {"xi1": ("S", "F"), "xi2": ("F", "F")}
    bc_u2 = {"xi1": ("F", "F"), "xi2": ("S", "F")}
    bc_u3 = {"xi1": ("S", "S"), "xi2": ("S", "S")}
    boundary_conditions = {"u1": bc_u1, "u2": bc_u2, "u3": bc_u3}

    expansion_size = {"u1": (6, 6), "u2": (6, 6), "u3": (6, 6)}
    displacement_field = EigenFunctionExpansion(expansion_size, rectangular_domain, boundary_conditions)

    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, 7850)
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, load)

    # =================================================================
    # 2. PASSO LINEAR (ESTADO FUNDAMENTAL U0)
    # =================================================================
    print("Calculando Estado Fundamental...")
    U1 = koiter_load_energy(shell)
    U2 = fast_koiter_quadratic_strain_energy(shell)

    F = tensor_derivative(U1, 0)
    K_E = tensor_derivative(tensor_derivative(U2, 0), 1)

    U0 = np.linalg.inv(K_E) @ F

    # =================================================================
    # 3. ANÁLISE DE FLAMBAGEM (K_G CORRIGIDA)
    # =================================================================
    print("Montando Matriz de Rigidez Geométrica...")
    # Lembre-se que sua função koiter_geometric_stiffness_matrix deve ter o fator 2.0 agora!
    K_G = koiter_geometric_stiffness_matrix(shell, U0)

    print("Resolvendo Autovalores...")
    # Resolvemos K_G * v = gamma * K_E * v  => lambda = 1/gamma
    gammas, eigenvectors = eigh(K_G, K_E)

    positive_gammas = gammas[gammas > 1e-12]
    gamma_max = np.max(positive_gammas)
    lambda_cr = 1.0 / gamma_max

    index_cr = np.where(gammas == gamma_max)[0][0]
    mode_cr = eigenvectors[:, index_cr]

    # =================================================================
    # 4. VERIFICAÇÃO CONTRA A LITERATURA
    # =================================================================
    D = (E * h ** 3) / (12 * (1 - nu ** 2))
    N_cr_analitico = (4 * np.pi ** 2 * D) / (a ** 2)

    print("\n" + "=" * 50)
    print("RESULTADO DA COMPARAÇÃO")
    print("=" * 50)
    print(f"Teoria de Timoshenko (N_cr): {N_cr_analitico:.2f} N/m")
    print(f"Resultado ShellPy (lambda) : {lambda_cr:.2f} N/m")
    print(f"Erro Relativo              : {abs(lambda_cr - N_cr_analitico) / N_cr_analitico * 100:.4f} %")
    print("=" * 50 + "\n")

    # =================================================================
    # 5. PLOTAGEM DO MODO DE INSTABILIDADE
    # =================================================================
    xi1_vals = np.linspace(0, a, 50)
    xi2_vals = np.linspace(0, b, 50)
    x, y = np.meshgrid(xi1_vals, xi2_vals, indexing='ij')

    mode1_cr = shell.displacement_expansion(mode_cr, x, y)
    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)
    mode = reciprocal_base[0] * mode1_cr[0] + reciprocal_base[1] * mode1_cr[1] + reciprocal_base[2] * mode1_cr[2]

    # Normalização para visualização
    mode_norm = mode / np.max(np.abs(mode)) * h * plot_scale
    z = shell.mid_surface_geometry(x, y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    norm = colors.Normalize(vmin=np.min(mode1_cr[2]), vmax=np.max(mode1_cr[2]))
    facecolors = plt.cm.jet(norm(mode1_cr[2]))

    ax.plot_surface(z[0, 0] + mode_norm[0], z[1, 0] + mode_norm[1], z[2, 0] + mode_norm[2],
                    facecolors=facecolors, edgecolor='k', linewidth=0.1, alpha=0.9)

    ax.set_title(f"Modo de Flambagem Crítico (N_cr = {lambda_cr:.2f} N/m)")
    plt.show()