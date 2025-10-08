import numpy as np


class ConstantThickness:
    def __init__(self, h):
        self._h = h

    def __call__(self, xi1=None, xi2=None):
        if xi1 is None:
            # Se não for fornecido xi1, retorna apenas o valor escalar
            return self._h

        # Converte xi1 em array para lidar com casos escalares e matriciais
        xi1 = np.asarray(xi1)
        return np.full_like(xi1, self._h, dtype=float)