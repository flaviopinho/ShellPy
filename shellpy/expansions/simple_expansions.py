import numpy as np


def fourier_expansion_for_periodic_solutions(edges, maximum_derivative=3, maximum_mode=3):
    expansion = {}
    L = edges[1] - edges[0]

    def func(n, derivative):
        k = (n+1) // 2  # Índice do harmônico
        omega = (2 * np.pi * k) / L  # Frequência angular
        phase_shift = (n % 2 + derivative) * (np.pi / 2)  # Mudança de fase
        return lambda x: omega ** derivative * np.cos(omega * x - phase_shift)

    for modo in range(maximum_mode):
        for derivative in range(maximum_derivative):
            expansion[(modo, derivative)] = func(modo, derivative)

    return expansion


def constant_value_expansion(maximum_derivative=3):
    expansion = {}
    for derivative in range(maximum_derivative):
        if derivative == 0:
            func = lambda x: np.ones(np.shape(x))
        elif derivative > 0:
            func = lambda x: np.zeros(np.shape(x))

        expansion[(0, derivative)] = func

    return expansion
