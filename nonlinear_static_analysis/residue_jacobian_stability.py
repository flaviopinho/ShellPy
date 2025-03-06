import numpy as np


def shell_residue(F_int, F_ext, x, *args):
    u = x[:-1]
    p = x[-1]
    F_int_tot = np.einsum('ij, j->i', F_int[0], u, optimize=True) + \
                np.einsum('ijk, j, k->i', F_int[1], u, u, optimize=True) + \
                np.einsum('ijkl, j, k, l->i', F_int[2], u, u, u, optimize=True)
    return F_int_tot + F_ext * p


def shell_jacobian(J_int, F_ext, x, *args):
    u = x[:-1]
    p = [-1]
    J_int_tot = J_int[0] + \
                np.einsum('ijk, k->ij', J_int[1], u, optimize=True) + \
                np.einsum('ijkl, k, l->ij', J_int[2], u, u, optimize=True)

    return np.hstack((J_int_tot, F_ext[:, np.newaxis]))


def shell_stability(u, J, model, *args):
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