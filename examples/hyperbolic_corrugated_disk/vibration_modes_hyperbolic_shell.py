import sympy as sym
import numpy as np
import pandas as pd  # <-- Adicionado para exportação de dados
import matplotlib.pyplot as plt
from scipy.linalg import eig, eigh
import seaborn as sns

from examples.hyperbolic_corrugated_disk.radial_localization_factor import radial_localization_factor, \
    simplified_radial_localization_factor
from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy import xi1_, xi2_, MidSurfaceGeometry
from shellpy import Shell
from shellpy.expansions.polinomial_expansion import LegendreSeries
from shellpy.materials.isotropic_homogeneous_linear_elastic_material import IsotropicHomogeneousLinearElasticMaterial
from shellpy.sanders_koiter import fast_koiter_kinetic_energy, fast_koiter_quadratic_strain_energy
from shellpy.tensor_derivatives import tensor_derivative
from shellpy import ConstantThickness

if __name__ == "__main__":
    # Mantendo seus parâmetros originais
    n = 10
    p = 10
    L = 1.0
    R_in = 0.3
    H = 0.3
    h = L / 100
    density = 1
    E = 1
    nu = 0.3

    mode_r = max(10, p * 3)
    mode_theta = max(10, n * 3)

    n_int_x = mode_r * 2
    n_int_y = mode_theta * 2
    n_int_z = 4

    rectangular_domain = RectangularMidSurfaceDomain(R_in, R_in + L, 0, 2 * np.pi)

    expansion_size = {"u1": (mode_r, mode_theta), "u2": (mode_r, mode_theta), "u3": (mode_r, mode_theta)}

    boundary_conditions = {
        "u1": {"xi1": ("S", "F"), "xi2": ("R", "R")},
        "u2": {"xi1": ("S", "F"), "xi2": ("R", "R")},
        "u3": {"xi1": ("C", "F"), "xi2": ("R", "R")}
    }

    displacement_field = LegendreSeries(expansion_size, rectangular_domain, boundary_conditions)

    R_ = sym.Matrix([
        xi1_ * sym.cos(xi2_),
        xi1_ * sym.sin(xi2_),
        H * ((xi1_ - R_in) / L) ** p * sym.cos(n * xi2_)
    ])

    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)
    material = IsotropicHomogeneousLinearElasticMaterial(E, nu, density)

    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, None)

    T = fast_koiter_kinetic_energy(shell, n_int_x, n_int_y, n_int_z)
    U2p = fast_koiter_quadratic_strain_energy(shell, n_int_x, n_int_y, n_int_z)

    M = tensor_derivative(tensor_derivative(T, 0), 1)
    K = tensor_derivative(tensor_derivative(U2p, 0), 1)

    n_modes_to_keep = 200
    eigen_vals, eigen_vectors = eigh(K, M, subset_by_index=(0, n_modes_to_keep - 1))
    sorted_indices = np.argsort(eigen_vals.real)
    eigen_vals = eigen_vals[sorted_indices]
    eigen_vectors = np.real(eigen_vectors[:, sorted_indices])

    omega = np.sqrt(eigen_vals.real)
    freq = omega / (2 * np.pi)

    # --- CÁLCULO DOS FATORES DE LOCALIZAÇÃO ---
    print("Calculando fatores de localização radial...")
    rlf_values = simplified_radial_localization_factor(shell, eigen_vectors, n_int_x, n_int_y)

    # --------------------------------------------------------------
    # EXPORTAÇÃO DOS DADOS PARA ARQUIVO
    # --------------------------------------------------------------
    print("Salvando resultados no arquivo...")

    # Criando um DataFrame com os Modos, Frequências e Índices de Localização
    df_results = pd.DataFrame({
        "Modo": np.arange(1, len(freq) + 1),
        "Frequencia_Hz": freq,
        "Fator_Localizacao_Radial": rlf_values
    })

    # Construindo o nome do arquivo com os parâmetros geométricos
    # O arquivo gerado terá um nome como: resultados_modais_n20_p10_L1.0_Rin0.3_H0.3_h0.01.csv
    filename = f"resultados_modais_shellpy_n{n}_p{p}_L{L}_Rin{R_in}_H{H}_h{h}.csv"

    # Salvando em CSV (usando ponto e vírgula como separador para facilitar abertura no Excel em pt-BR)
    df_results.to_csv(filename, index=False, sep=";", decimal=",")
    print(f"Arquivo salvo com sucesso: {filename}")

    # DICA: Se preferir salvar direto em .xlsx nativo do Excel, você pode usar:
    # df_results.to_excel(filename.replace('.csv', '.xlsx'), index=False)

    # --------------------------------------------------------------
    # Frequência vs Radial Localization Factor
    # --------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(freq, rlf_values, c=rlf_values, cmap='coolwarm', edgecolors='k', alpha=0.7)
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Global')
    plt.axhline(y=0.85, color='red', linestyle=':', label='Edge Localized')

    plt.xlabel("Frequência (Hz)")
    plt.ylabel(r"Radial Localization Factor ($\eta$)")
    plt.title(f"Localização Modal vs Frequência (p={p}, n={n})")
    plt.colorbar(label=r'$\eta$ intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --------------------------------------------------------------
    # Gráfico de Densidade Acumulada (Igual ao seu)
    # --------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 7))
    sns.histplot(freq, bins='fd', kde=True, color='royalblue', stat="density", ax=ax1, alpha=0.3,
                 label='Densidade (KDE)')
    ax1.set_xlabel("Frequência (Hz)")
    ax1.set_ylabel("Densidade", color='royalblue')
    ax2 = ax1.twinx()
    ax2.step(freq, np.arange(1, len(freq) + 1), where='post', color='darkorange', linewidth=2, label='N(f) Acumulado')
    ax2.set_ylabel("N(f)", color='darkorange')
    plt.title(f"Análise Espectral: Casca Corrugada")
    plt.show()