import numpy as np
from shellpy import cache_function
from shellpy.numeric_integration.default_integral_division import n_integral_default_x


@cache_function
def gauss_weights_simple_integral(edges, n_x=n_integral_default_x, n_gauss_points=4):
    """
    Função de integração usando quadratura de Gauss-Legendre,
    dividindo o intervalo em n_x partes e usando n_gauss_points pontos de Gauss em cada parte.

    :param edges: Uma tupla definindo as bordas do domínio Exemplo (-h/2, h/2).
    :param n_x: O número de subdivisões.
    :param n_gauss_points: O número de pontos de Gauss por subdivisão.
    :return: Os pontos de integração (x) e os pesos correspondentes (W).
    """
    # Obter pontos e pesos de Gauss-Legendre para o intervalo [-1, 1]
    gauss_points, gauss_weights = np.polynomial.legendre.leggauss(n_gauss_points)

    # Calcular o tamanho de cada subdivisão
    a, b = edges
    h = (b - a) / n_x

    # Inicializar listas
    shape = np.shape(a) + (n_x * n_gauss_points,)
    x_points = np.zeros(shape)
    W = np.zeros_like(x_points)

    # Para cada subdivisão
    for i in range(n_x):
        # Definir os limites da subdivisão atual
        x_left = a + i * h
        x_right = a + (i + 1) * h

        # Transformar pontos de Gauss do intervalo [-1, 1] para [x_left, x_right]
        x_local = np.multiply.outer((x_right - x_left) / 2, gauss_points) + np.expand_dims((x_right + x_left) / 2, axis=-1)

        # Calcular pesos para esta subdivisão
        W_local = np.multiply.outer((x_right - x_left) / 2, gauss_weights).squeeze()

        # Adicionar pontos e pesos às listas
        idx = slice(i * n_gauss_points, (i + 1) * n_gauss_points)
        x_points[..., idx] = x_local  # adiciona dimensão para broadcast
        W[..., idx] = W_local

    return x_points, W
