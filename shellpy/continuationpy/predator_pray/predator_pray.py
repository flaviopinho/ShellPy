import matplotlib.pyplot as plt
import numpy as np

from ..continuation import Continuation


def predator_pray_residue(u, *args):
    n = 2
    o = 3

    x = u[:n]
    p = u[n:n + o]

    F = np.array([p[0] * x[0] * (1 - x[0]) - x[0] * x[1] - p[2] * (1 - np.exp(-p[1] * x[0])),
                  -x[1] + p[0] * x[0] * x[1]])

    g = np.array([p[0] - 3, p[1] - 5])

    H = np.concatenate((F, g))

    return H


def predator_pray_jacobian(u, *args):
    n = 2
    o = 3

    x = u[:n]
    p = u[n:n + o]

    dFdx = np.array([[p[0] * (1 - x[0]) - p[0] * x[0] - x[1] - p[2] * p[1] * np.exp(-p[1] * x[0]), -x[0]],
                     [p[0] * x[1], p[0] * x[0] - 1]])
    dFdp = np.array([[x[0] * (1 - x[0]), -p[2] * x[0] * np.exp(-p[1] * x[0]), -(1 - np.exp(-p[1] * x[0]))],
                     [x[0] * x[1], 0, 0]])

    dgdx = np.zeros((n, n))
    dgdp = np.array([[1, 0, 0],
                     [0, 1, 0]])

    J = np.block([[dFdx, dFdp], [dgdx, dgdp]])

    return J


def stability_fixed_point(u, J, model, *args):
    # Extrai a submatriz Jx
    Jx = J[:modelo['n'], :modelo['n']]

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


if __name__ == "__main__":
    # Numero de variaveis
    n = 2
    # Numero de parametros
    p = 3

    # Limites de interesse das variaveis e parametros
    boundary = np.array([[-1, 2],
                         [-1, 3],
                         [2.9, 3.1],
                         [4.9, 5.1],
                         [0, 1]])
    # Definindo modelo
    modelo = {'n': n,
              'p': p,
              'residue': predator_pray_residue,
              'jacobian': predator_pray_jacobian,
              'stability_check': stability_fixed_point,
              'boundary': boundary}

    continuation = Continuation(modelo)


    # Determinacao de um ponto regular inicial
    u0 = np.array([1 / 3, 2, 3, 5, 0])
    H0 = continuation.model['residue'](u0)
    J0 = continuation.model['jacobian'](u0)
    t0 = continuation.tangent_vector(J0)
    w0 = 1

    continuation.continue_branch(u0, t0, w0, 'Branch1')

    for point_data in continuation.branches['Branch1']:
        if point_data["point_type"] == "BPS":
            u = point_data['u']  # Retorna o primeiro BPS encontrado
            tu = point_data['t_u']

    t1, t2 = continuation.branch_switching(u)
    print(np.dot(t1, tu), np.dot(t2, tu))
    if abs(np.dot(t1, tu)) < abs(np.dot(t2, tu)):
        tb = t1
    else:
        tb = t2

    continuation.continue_branch(u, t1, -w0, 'Branch2')
    continuation.continue_branch(u, t1, w0, 'Branch3')

    for point_data in continuation.branches['Branch3']:
        if point_data["point_type"] == "BPS":
            u = point_data['u']  # Retorna o primeiro BPS encontrado
            tu = point_data['t_u']

    J = continuation.model['jacobian'](u)
    t1, t2 = continuation.branch_switching(u)
    print(np.dot(t1, tu), np.dot(t2, tu))
    if abs(np.dot(t1, tu)) < abs(np.dot(t2, tu)):
        tb = t1
    else:
        tb = t2

    continuation.continue_branch(u, t1, w0, 'Branch4')
    continuation.continue_branch(u, -t1, w0, 'Branch5')

    plt.show()
