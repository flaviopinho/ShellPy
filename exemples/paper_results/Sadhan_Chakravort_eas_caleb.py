# Análise de casca canoide parabólica ortotrópica laminada

import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
from scipy.linalg import eigh

from shellpy.expansions.enriched_cosine_expansion import EnrichedCosineExpansion
from shellpy.expansions.polinomial_expansion import LegendreSeries
from shellpy.fsdt7_eas.mass_matrix import mass_matrix
from shellpy.fsdt7_eas.stiffness_matrix import stiffness_matrix
from shellpy.materials.laminate_orthotropic_material import Lamina, LaminateOrthotropicMaterial
from shellpy import ConstantThickness, MidSurfaceGeometry, RectangularMidSurfaceDomain, Shell, xi1_, xi2_
from shellpy.displacement_expansion import (
    SSSS_fsdt6, CCCC_fsdt6, CCFF_fsdt6, SSCC_fsdt6,
    CCSS_fsdt6, CSCS_fsdt6, SCSC_fsdt6, FFCC_fsdt6,
    CFCF_fsdt6, FCFC_fsdt6
)

if __name__ == "__main__":
    # Parâmetros de integração
    integral_x = 30
    integral_y = 30
    integral_z = 15

    # Configuração geométrica - Dados da Tabela 1
    a = 1.0       # Comprimento
    b = 1.0       # Largura
    h = a / 100   # Espessura
    hh = a / 2.5  # Altura máxima
    h_l = hh * 0.25  # Altura mínima
    
    # Parâmetros do conoide
    f1 = h_l
    f2 = hh

    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    # Definição da geometria do conoide parabólico
    Z_conoid = f1 * (1 - (1 - f2/f1) * xi1_ / a) * (1 - (2 * xi2_ / b - 1)**2)
    R_ = sym.Matrix([xi1_, xi2_, Z_conoid])

    mid_surface_geometry = MidSurfaceGeometry(R_)
    thickness = ConstantThickness(h)

    # Propriedades do material
    E22 = 1.0
    density = 1.0
    E11 = 25.0 * E22
    E33 = E22
    G12 = 0.5 * E22
    G13 = 0.5 * E22
    G23 = 0.2 * E22
    nu12 = 0.25
    nu13 = 0.25
    nu23 = 0.25


    # Criar lâminas
    t_lamina = h/8
    def create_lamina(angle_deg):
        """Função auxiliar para criar uma lâmina"""
        return Lamina(
            E_11=E11,
            E_22=E22,
            E_33=E33,
            nu_12=nu12,
            nu_13=nu13,
            nu_23=nu23,
            G_12=G12,
            G_13=G13,
            G_23=G23,
            density=density,
            angle=angle_deg * np.pi / 180.0,  # converter graus para radianos
            thickness=t_lamina,
        )

    # Sequência em graus
    angles = [0, 90, 0, 90, 0, 90, 0, 90]
    laminas = [create_lamina(angle) for angle in angles]

    material = LaminateOrthotropicMaterial(laminas, thickness)

    # Expansões e condição de contorno
    n_modos = 20

    expansion_size = {k: (n_modos, n_modos) for k in ("u1", "u2", "u3", "v1", "v2", "v3")}
    displacement_field = EnrichedCosineExpansion(expansion_size, rectangular_domain, SSSS_fsdt6)

    # Enhanced Assumed Strain (EAS)
    eas_field = LegendreSeries({"u1": (n_modos, n_modos)}, rectangular_domain,
                               {"u1": {"xi1": ("F", "F"), "xi2": ("F", "F")}})

    # Shell: monta o modelo (geometria, espessura, domínio, material, expansão de deslocamento,EAS).
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, None)

    # Visualização da geometria
    print("Visualização da geometria...")
    xi1_g = np.linspace(*rectangular_domain.edges["xi1"], 100)
    xi2_g = np.linspace(*rectangular_domain.edges["xi2"], 100)
    xg, yg = np.meshgrid(xi1_g, xi2_g, indexing="ij")
    zg = shell.mid_surface_geometry(xg, yg)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(9, 7))
    ax.plot_surface(zg[0, 0], zg[1, 0], zg[2, 0], cmap="viridis", alpha=0.95, antialiased=True,
                    rstride=2, cstride=2, edgecolor="0.15", lw=0.2)

    ax.set_title("Casca canoide laminada", fontsize=13, fontweight="bold")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f"get_{e}lim")() for e in "xyz")])
    plt.tight_layout()
    plt.show()

    # Autovalores: M e K → ω (frequências naturais) - K·φ = λ·M·φ; λ = ω²
    M = mass_matrix(shell, integral_x, integral_y, integral_z)
    K = stiffness_matrix(shell, eas_field, integral_x, integral_y, integral_z)
    eigen_vals, eigen_vectors = eigh(K, M)  # eigh para matrizes simétricas; retorna λ e φ
    omega = np.sqrt(eigen_vals)  # ω = √λ [rad/s]

    # Filtra espúrios: |Re(ω)| > tol, mantém parte real, ordena por ω
    mask = np.isfinite(omega) & (np.abs(np.real(omega)) > 1e-2)
    omega, eigen_vectors = omega[mask].real, np.real(eigen_vectors[:, mask])
    idx = np.argsort(omega)
    omega, eigen_vectors = omega[idx], eigen_vectors[:, idx]

    # Número de modos a plotar
    n_modes = 5

    # Frequências naturais em Hz 
    freq_Hz = omega / (2 * np.pi)  # Conversão rad/s → Hz

    # frequência adimensional (pg. 1414 do artigo)
    omega_bar = omega * (a ** 2) * np.sqrt(density / (E22 * h ** 2))

    print(f"\nFrequências naturais (Hz):\n{freq_Hz[:n_modes]}")
    print(f"\nω̄ adimensional:\n{omega_bar[:n_modes]}")

    # --- Modos de vibração ---
    xi1 = np.linspace(*rectangular_domain.edges["xi1"], 100)
    xi2 = np.linspace(*rectangular_domain.edges["xi2"], 100)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')
    z = shell.mid_surface_geometry(x, y)

    reciprocal_base = shell.mid_surface_geometry.reciprocal_base(x, y)

    fig, axes = plt.subplots(1, n_modes, figsize=(15, 5), subplot_kw={"projection": "3d"}, constrained_layout=True)
    scmap = plt.cm.ScalarMappable(cmap="jet")

    for i in range(n_modes):
        # Deslocamento covariante (u1,u2,u3) e projeção em cartesianas
        mode1 = shell.displacement_expansion(eigen_vectors[:, i], x, y)
        mode = reciprocal_base[0] * mode1[0] + reciprocal_base[1] * mode1[1] + reciprocal_base[2] * mode1[2]
        mode = mode / np.max(np.abs(mode)) * h  # normaliza |max| e escala por h

        ax = axes[i]
        ax.plot_surface(z[0, 0] + mode[0], z[1, 0] + mode[1], z[2, 0] + mode[2],
                        facecolors=scmap.to_rgba(mode1[2]), edgecolor="black", linewidth=0.1)
        ax.set_title(f"Modo {i + 1} = {omega_bar[i]:.2f}", fontsize=11)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")])

    plt.show()
