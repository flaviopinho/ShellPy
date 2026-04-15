import numpy as np
from scipy.optimize import nnls
from scipy.stats import qmc
from typing import Tuple, List, Union
import matplotlib.pyplot as plt


def empirical_cubature_method(
        linear_strains: List[np.ndarray],
        nonlinear_strains: List[np.ndarray],
        constitutive_matrices: List[np.ndarray],
        original_weights: np.ndarray,
        num_samples: int = 100,
        max_amplitude: Union[float, np.ndarray] = 1,
        seed: int = 42,
        weight_threshold: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main wrapper for the Empirical Cubature Method (ECM) applied to HSDT shell elements.

    Args:
        linear_strains: List [E0_l, E1_l, E2_l] of linear strain tensors.
        nonlinear_strains: List [E0_nl, E1_nl, E2_nl] of nonlinear strain tensors.
        constitutive_matrices: List [C0, C1, C2, C3, C4] of integrated constitutive matrices.
        original_weights: Array of original Gaussian integration weights (nx, ny).
        num_samples: Number of LHS snapshots to generate.
        max_amplitude: Maximum amplitude for the Ritz coefficients.
        seed: Random seed for LHS reproducibility.
        weight_threshold: Threshold below which an optimized weight is considered zero.

    Returns:
        active_indices: 1D array of selected integration point indices.
        reduced_weights: 1D array of new optimized weights for the selected points.
    """
    # Extract the number of degrees of freedom from the first linear strain tensor
    # Assuming shape is (dof_count, 6, nx, ny)
    dof_count = linear_strains[0].shape[0]

    # Step 1: Generate statistical displacement states (snapshots)
    u_samples = generate_lhs_displacement_samples(dof_count, num_samples, max_amplitude, seed)

    # Step 2: Build the snapshot matrix (A) and the exact integral vector (b)
    A_matrix, b_target = build_ecm_snapshot_matrix(
        u_samples, linear_strains, nonlinear_strains, constitutive_matrices, original_weights
    )

    # Step 3: Run the Non-Negative Least Squares (NNLS) optimization to sparsify points
    active_indices, reduced_weights = optimize_cubature_points_nnls(A_matrix, b_target, weight_threshold)

    return active_indices, reduced_weights


def generate_lhs_displacement_samples(
        dof_count: int,
        num_samples: int,
        max_amplitude: Union[float, np.ndarray] = 0.01,
        seed: int = 42
) -> np.ndarray:
    """
    Generates representative displacement samples (Ritz coefficients) using Latin Hypercube Sampling.
    """
    # Initialize the LHS engine for a 'dof_count'-dimensional space
    sampler = qmc.LatinHypercube(d=dof_count, seed=seed)

    # Generate samples in the unit hypercube [0, 1]
    sample_unit = sampler.random(n=num_samples)

    # Scale to the physical interval [-max_amplitude, max_amplitude]
    # This ensures sampling of both positive and negative deformations
    if np.isscalar(max_amplitude):
        lower_bounds = np.full(dof_count, -max_amplitude)
        upper_bounds = np.full(dof_count, max_amplitude)
    else:
        lower_bounds = -np.array(max_amplitude)
        upper_bounds = np.array(max_amplitude)

    u_samples = qmc.scale(sample_unit, lower_bounds, upper_bounds)

    return u_samples


def build_ecm_snapshot_matrix(
        u_samples: np.ndarray,
        linear_strains: List[np.ndarray],
        nonlinear_strains: List[np.ndarray],
        constitutive_matrices: List[np.ndarray],
        original_weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Avalia os esforços seccionais para todos os snapshots usando vetorização completa
    para altíssima performance.
    """
    num_samples = u_samples.shape[0]
    num_points = original_weights.size

    print("Coletando snapshots (Vetorização Completa Ativada)...")

    # Flag fundamental para encontrar o caminho mais rápido de contração de tensores
    opt = True

    # --------------------------------------------------------------------------------
    # 1. Deformações Lineares para TODOS os snapshots simultaneamente
    # linear_strains shape: (dof, 6, x, y) -> índice 'iaxy'
    # u_samples shape: (samples, dof) -> índice 'ki'
    # Resultado shape: (samples, 6, x, y) -> índice 'kaxy'
    # --------------------------------------------------------------------------------
    E0_l = np.einsum('iaxy, ki -> kaxy', linear_strains[0], u_samples, optimize=opt)
    E1_l = np.einsum('iaxy, ki -> kaxy', linear_strains[1], u_samples, optimize=opt)
    E2_l = np.einsum('iaxy, ki -> kaxy', linear_strains[2], u_samples, optimize=opt)

    # --------------------------------------------------------------------------------
    # 2. Deformações Não-Lineares (Pre-calculamos u_i * u_j para economizar tempo)
    # uu shape: (samples, dof, dof) -> índice 'kij'
    # --------------------------------------------------------------------------------
    uu = np.einsum('ki, kj -> kij', u_samples, u_samples, optimize=opt)

    E0_nl = np.einsum('ijaxy, kij -> kaxy', nonlinear_strains[0], uu, optimize=opt)
    E1_nl = np.einsum('ijaxy, kij -> kaxy', nonlinear_strains[1], uu, optimize=opt)
    E2_nl = np.einsum('ijaxy, kij -> kaxy', nonlinear_strains[2], uu, optimize=opt)

    # Deformações Totais: (samples, 6, x, y)
    E0 = E0_l + E0_nl
    E1 = E1_l + E1_nl
    E2 = E2_l + E2_nl

    # --------------------------------------------------------------------------------
    # 3. Integrandos dos Esforços Seccionais
    # C_mat shape: (6, 6, x, y) -> índice 'abxy'
    # E shape: (samples, 6, x, y) -> índice 'kbxy'
    # Resultado shape: (samples, 6, x, y) -> índice 'kaxy'
    # --------------------------------------------------------------------------------
    C0, C1, C2, C3, C4 = constitutive_matrices

    L0_field = (np.einsum('abxy, kbxy -> kaxy', C0, E0, optimize=opt) +
                np.einsum('abxy, kbxy -> kaxy', C1, E1, optimize=opt) +
                np.einsum('abxy, kbxy -> kaxy', C2, E2, optimize=opt))

    L1_field = (np.einsum('abxy, kbxy -> kaxy', C1, E0, optimize=opt) +
                np.einsum('abxy, kbxy -> kaxy', C2, E1, optimize=opt) +
                np.einsum('abxy, kbxy -> kaxy', C3, E2, optimize=opt))

    L2_field = (np.einsum('abxy, kbxy -> kaxy', C2, E0, optimize=opt) +
                np.einsum('abxy, kbxy -> kaxy', C3, E1, optimize=opt) +
                np.einsum('abxy, kbxy -> kaxy', C4, E2, optimize=opt))

    # --------------------------------------------------------------------------------
    # 4. Achatamento e Montagem Mágica da Matriz A
    # L_stacked junta L0, L1, L2 num shape: (samples, 3, 6, x, y)
    # --------------------------------------------------------------------------------
    L_stacked = np.stack([L0_field, L1_field, L2_field], axis=1)

    # Remodelamos para (samples, 18 componentes, num_points)
    L_reshaped = L_stacked.reshape(num_samples, 18, num_points)

    # Achata as duas primeiras dimensões para resultar em (18 * samples, num_points)
    A_matrix = L_reshaped.reshape(num_samples * 18, num_points)

    # --------------------------------------------------------------------------------
    # 5. Cálculo Exato do Alvo (b_target)
    # Uma única multiplicação de matriz por vetor substitui o loop inteiro!
    # --------------------------------------------------------------------------------
    w_flat = original_weights.flatten()
    b_target = A_matrix @ w_flat

    # --------------------------------------------------------------------------------
    # 6. FILTRO DE ZEROS
    # --------------------------------------------------------------------------------
    non_zero_indices = np.where(np.any(np.abs(A_matrix) > 1e-15, axis=1))[0]
    A_filtered = A_matrix[non_zero_indices, :]
    b_filtered = b_target[non_zero_indices]

    active_components = len(non_zero_indices) // num_samples
    print(f"Componentes de esforço ativas detectadas: {active_components} de 18.")

    return A_filtered, b_filtered


def optimize_cubature_points_nnls(
        A_matrix: np.ndarray,
        b_target: np.ndarray,
        weight_threshold: float = 1e-12,
        svd_tolerance: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve o problema de otimização ECM usando compressão SVD de alta performance.
    """
    num_eqs, num_points = A_matrix.shape
    print(f"\nIniciando otimização para {num_points} pontos candidatos...")
    print(f"Tamanho original do sistema: {num_eqs} equações.")

    # ================================================================================
    # 1. ESCALONAMENTO (Pre-conditioning)
    # Garante que flexão e membrana tenham o mesmo "peso" para o otimizador
    # ================================================================================
    row_norms = np.linalg.norm(A_matrix, axis=1)
    row_norms[row_norms < 1e-15] = 1.0  # Proteção contra divisão por zero

    A_scaled = A_matrix / row_norms[:, np.newaxis]
    b_scaled = b_target / row_norms

    # ================================================================================
    # 2. COMPRESSÃO SVD (A Mágica da Velocidade)
    # Extrai as matrizes U (vetores singulares esquerdos) e S (valores singulares)
    # ================================================================================
    print("Calculando SVD para compressão do sistema (removendo redundâncias)...")
    U, S, Vt = np.linalg.svd(A_scaled, full_matrices=False)

    # Conta quantos modos (equações) realmente importam para a física
    # Ignoramos tudo que for menor que 10^-8 em relação ao maior valor singular
    k_modes = np.sum(S > svd_tolerance * S[0])
    print(f"Sucesso! SVD reduziu as equações de {num_eqs} para apenas {k_modes} modos dominantes.")

    # Projeta o sistema escalonado no novo espaço reduzido (multiplica por U transposto)
    U_trunc = U[:, :k_modes]
    A_comp = U_trunc.T @ A_scaled
    b_comp = U_trunc.T @ b_scaled

    # ================================================================================
    # 3. NNLS NO SISTEMA COMPRIMIDO
    # ================================================================================
    limite_iteracoes = 20 * num_points  # Limite estendido por segurança
    print(f"Rodando NNLS no sistema super-comprimido (maxiter={limite_iteracoes})...")

    new_weights, residual = nnls(A_comp, b_comp, maxiter=limite_iteracoes)

    # ================================================================================
    # 4. EXTRAÇÃO DOS PONTOS E DIAGNÓSTICO
    # ================================================================================
    active_indices = np.where(new_weights > weight_threshold)[0]
    reduced_weights = new_weights[active_indices]

    # Para ser honesto com a métrica, calculamos o erro no sistema REAL (não comprimido)
    forca_reconstruida = A_matrix[:, active_indices] @ reduced_weights
    erro_absoluto = np.linalg.norm(forca_reconstruida - b_target)
    target_norm = np.linalg.norm(b_target)
    erro_relativo = erro_absoluto / target_norm if target_norm > 0 else erro_absoluto

    print("-" * 50)
    print("Otimização ECM Concluída com Sucesso!")
    print(f"Pontos selecionados: {len(active_indices)} de {num_points} originais.")
    print(f"Redução de Malha: {(1 - len(active_indices) / num_points) * 100:.2f}%")
    print(f"Erro residual relativo (Física Real): {erro_relativo:.2e}")
    print("-" * 50)

    return active_indices, reduced_weights


def plot_ecm_points(xi1_matrix, xi2_matrix, active_indices):
    """
    Plota a malha original de pontos de Gauss e destaca os pontos
    selecionados pelo método de hiper-redução (ECM).

    Args:
        xi1_matrix (np.ndarray): Matriz de coordenadas xi1 originais (n_x, n_y).
        xi2_matrix (np.ndarray): Matriz de coordenadas xi2 originais (n_x, n_y).
        active_indices (np.ndarray): Índices 1D dos pontos que sobreviveram ao ECM.
    """
    # 1. "Achatamos" as matrizes originais para bater com os índices 1D do ECM
    xi1_flat = xi1_matrix.flatten()
    xi2_flat = xi2_matrix.flatten()

    # 2. Extraímos as coordenadas apenas dos pontos ativos
    xi1_ativos = xi1_flat[active_indices]
    xi2_ativos = xi2_flat[active_indices]

    # 3. Configuração do Gráfico
    plt.figure(figsize=(8, 8))

    # Plota a grade densa original (em cinza claro, menores)
    plt.scatter(xi1_flat, xi2_flat,
                c='lightgray', s=15, alpha=0.6,
                label=f'Grade Original de Gauss ({len(xi1_flat)} pts)')

    # Plota os sobreviventes do ECM (em vermelho, maiores, com borda)
    plt.scatter(xi1_ativos, xi2_ativos,
                c='crimson', s=70, edgecolor='black', zorder=5,
                label=f'Pontos ECM Selecionados ({len(active_indices)} pts)')

    # Estética
    plt.title('Redução de Cubatura Empírica (ECM) no Domínio Paramétrico', fontsize=14, pad=15)
    plt.xlabel(r'Coordenada paramétrica $\xi_1$', fontsize=12)
    plt.ylabel(r'Coordenada paramétrica $\xi_2$', fontsize=12)

    # Força a proporção 1:1 para não distorcer o domínio
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.show()