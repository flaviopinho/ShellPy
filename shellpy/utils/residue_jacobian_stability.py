import numpy as np


def shell_residue(F_int, F_ext, x, *args):
    u = x[:-1]
    p = x[-1]

    index_labels = "abcdefghijklmnopqrstuvwxyz"

    F_int_tot = sum(
        np.einsum(
            f"i{index_labels[:len(t.shape) - 1]},{','.join(index_labels[:len(t.shape) - 1])}->i",
            t, *[u] * (len(t.shape) - 1), optimize=True
        )
        for t in F_int
    )

    return F_int_tot + F_ext * p


def shell_jacobian(J_int, F_ext, x, *args):
    u = x[:-1]

    index_labels = "abcdefghijklmnopqrstuvwxyz"

    J_int_tot = J_int[0] + sum(
        np.einsum(
            f"ij{index_labels[:len(t.shape) - 2]},{','.join(index_labels[:len(t.shape) - 2])}->ij",
            t, *[u] * (len(t.shape) - 2), optimize=True
        )
        for t in J_int[1:]
    )

    return np.hstack((J_int_tot, F_ext[:, np.newaxis]))


def shell_stability(u, J, model, *args):
    # Extrai a submatriz estrutural Jx
    Jx = -J[:model['n'], :model['n']]

    # Determinação dos autovalores de Jx
    eigen_values = np.linalg.eigvals(Jx)

    # Partes reais e imaginárias dos autovalores
    real_part = np.real(eigen_values)
    imaginary_part = np.imag(eigen_values)

    # Estabilidade: a maior parte real ditará o ponto crítico de cruzamento do eixo
    stability = np.max(real_part)

    # --- DEFINIÇÃO DE TOLERÂNCIAS ---
    # Valores abaixo dessa tolerância são considerados zero (ruído numérico)
    tol_real = 1e-8
    tol_imag = 1e-8

    # Análise do tipo de bifurcação (usando a tolerância)
    index_real_positivo = real_part > tol_real
    num_real_positivo = np.sum(index_real_positivo)

    # Verifica se os autovalores instáveis (positivos) possuem parte imaginária significativa
    tem_parte_imaginaria = np.any(np.abs(imaginary_part[index_real_positivo]) > tol_imag)

    # Classificação
    if num_real_positivo == 2 and tem_parte_imaginaria:
        tipo = 'H'  # Hopf (Surgem em pares complexos conjugados)
    elif num_real_positivo == 1:
        # Ponto de sela (Saddle-Node) ou Bifurcação tipo Pitchfork (autovalor puramente real)
        tipo = 'SN'
    elif num_real_positivo == 0:
        tipo = 'PR'  # Ponto regular (totalmente estável)
    else:
        tipo = 'BC'  # Bifurcação complexa / Múltipla instabilidade

    return stability, tipo