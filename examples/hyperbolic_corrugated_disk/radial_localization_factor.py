import numpy as np

from shellpy import Shell
from shellpy.numeric_integration.default_integral_division import n_integral_default_x, n_integral_default_y
from shellpy.numeric_integration.gauss_integral import gauss_weights_simple_integral
from shellpy.numeric_integration.integral_weights import double_integral_weights


def radial_localization_factor(shell: Shell,
                               u,
                               n_x=n_integral_default_x,
                               n_y=n_integral_default_y,
                               integral_method=gauss_weights_simple_integral):

    # 1. Obter pesos e pontos (Fixos para a geometria)
    xi1, xi2, Wxy = double_integral_weights(shell.mid_surface_domain, n_x, n_y, integral_method)

    # 2. Geometria da Casca (Fixa)
    G = shell.mid_surface_geometry.metric_tensor_covariant_components_extended(xi1, xi2)
    sqrtG = shell.mid_surface_geometry.sqrtG(xi1, xi2)

    # 3. Pré-calcular Funções de Forma para todos os DOFs em todos os pontos de integração
    # Shape: (n_dofs, n_campos, n_points_x, n_points_y)
    n_dofs = shell.displacement_expansion.number_of_degrees_of_freedom()
    n_fields = shell.displacement_expansion.number_of_fields()

    phi = np.zeros((n_dofs, n_fields, xi1.shape[0], xi1.shape[1]))
    for n in range(n_dofs):
        phi[n] = shell.displacement_expansion.shape_function(n, xi1, xi2)

    # 4. Cálculo do Kernel de Integração
    # r_relative para o numerador
    xi1_min, xi1_max = shell.mid_surface_domain.edges["xi1"]
    r_rel = (xi1 - xi1_min) / (xi1_max - xi1_min)

    # Peso comum: W * sqrt(G)
    dV = Wxy * sqrtG

    # Construção da matriz M (Kernel do Denominador) e Mn (Kernel do Numerador)
    # Usamos einsum para contrair os campos (i, j) e os pontos de integração (x, y)
    # 'a' e 'b' são os índices dos DOFs
    M = np.einsum('aixy, ijxy, bjxy, xy -> ab', phi, G, phi, dV, optimize=True)
    Mn = np.einsum('aixy, ijxy, bjxy, xy -> ab', phi, G, phi, dV * r_rel, optimize=True)

    # Se u for um único vetor, garante que seja 2D para a conta matricial
    if u.ndim == 1:
        u = u[np.newaxis, :]

    # Numerador: u^T * Mn * u
    # Denominador: u^T * M * u
    # Fazemos isso para várias colunas/casos de uma vez
    num = np.sum(u * (Mn @ u), axis=0)
    den = np.sum(u * (M @ u), axis=0)

    return num / den