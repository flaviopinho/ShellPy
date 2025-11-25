import numpy as np


class LaminateOrthotropicMaterial:
    def __init__(self, laminas, thickness):
        """
        laminas: lista de objetos Lamina (de baixo para cima)
        """
        self.laminas = laminas
        self.thickness = thickness

    def _compute_average_density(self):
        """Massa por unidade de área do laminado."""
        return sum(l.density * l.thickness for l in self.laminas) * self.total_thickness

    def lamina_index(self, xi1, xi2, xi3):
        """
        Retorna o índice da lâmina correspondente a cada ponto (xi1, xi2, xi3).

        - xi1, xi2 podem ser escalares ou arrays compatíveis.
        - xi3 pode ser escalar, 1D (n,) ou já ter shape xi1.shape + (n,).
        - O retorno tem o mesmo shape de xi3_exp (xi3 broadcastado para xi1.shape + (n,)).
        """

        xi3 = np.asarray(xi3)

        # Broadcast de xi3 para shape (xi1.shape + (n,)) se necessário
        if xi3.shape[:-1] != np.shape(xi1):
            xi3_exp = np.broadcast_to(xi3, np.shape(xi1) + (xi3.shape[-1],))
        else:
            xi3_exp = xi3

        # Espessura total física local (shape == np.shape(xi1))
        t_total = np.asarray(self.thickness(xi1, xi2))

        if np.any(t_total <= 0):
            raise ValueError("A espessura total deve ser positiva em todos os pontos.")

        # Normaliza xi3 para o intervalo [-1, 1]
        t_total_exp = t_total[..., np.newaxis]  # shape (..., 1)
        xi3_norm = 2.0 * xi3_exp / t_total_exp

        # Limites superiores normalizados de cada lâmina
        z = -1.0
        z_tops = []
        for l in self.laminas:
            z += l.thickness
            z_tops.append(z)
        z_tops = np.array(z_tops)

        # Inicializa saída (com -1 para detectar fora do intervalo)
        index = np.full_like(xi3_norm, fill_value=-1, dtype=int)

        # Preenche o índice da lâmina
        z_low = -1.0
        for i, z_top in enumerate(z_tops):
            if i < len(z_tops) - 1:
                mask = (xi3_norm >= z_low) & (xi3_norm < z_top)
            else:
                mask = (xi3_norm >= z_low) & (xi3_norm <= z_top)
            index[mask] = i
            z_low = z_top

        # Verifica se há pontos fora do intervalo [-1, 1]
        if np.any(index == -1):
            raise ValueError("Alguns valores de xi3 estão fora do intervalo da espessura total do laminado.")

        return index

    def angle(self, xi1, xi2, xi3):
        """
        Retorna o ângulo da lâmina correspondente a cada ponto (xi1, xi2, xi3).

        - xi1, xi2 podem ser escalares ou arrays compatíveis.
        - xi3 pode ser escalar ou array multidimensional.
        - O retorno tem o mesmo shape de xi3 broadcastado para xi1.shape + (n,).
        """
        # Obtém o índice da lâmina para cada xi3
        indices = self.lamina_index(xi1, xi2, xi3)  # shape igual ao xi3 broadcastado

        # Array de ângulos das lâminas
        angles = np.array([l.angle for l in self.laminas])

        # Retorna o ângulo correspondente a cada ponto
        return angles[indices]

    def orthotropic_voigt_matrix(self, xi1, xi2, xi3):
        """
        Retorna a matriz constitutiva ortotrópica (6x6) para cada ponto xi3.
        Output: ndarray com shape (6,6) + xi3.shape
        """
        # Obtém o índice da lâmina correspondente a cada ponto
        index = self.lamina_index(xi1, xi2, xi3)  # shape = xi3 broadcastado

        # Shape final do array de saída
        shape_out = (6, 6) + np.shape(index)
        C_all = np.zeros(shape_out, dtype=float)

        # Monta as matrizes C para cada lâmina
        C_laminas = []
        for lamina in self.laminas:
            E1, E2, E3 = lamina.E_11, lamina.E_22, lamina.E_33
            nu12 = lamina.nu_12
            nu13 = lamina.nu_13
            nu23 = lamina.nu_23

            G12 = lamina.G_12
            G13 = lamina.G_13
            G23 = lamina.G_23

            nu21 = nu12 * E2 / E1
            nu31 = nu13 * E3 / E1
            nu32 = nu23 * E3 / E2

            C = np.zeros((6, 6))
            C[0, 0] = (1 / E1)
            C[0, 1] = -nu21 / E2
            C[0, 2] = -nu31 / E3

            C[1, 0] = -nu12 / E1
            C[1, 1] = 1 / E2
            C[1, 2] = -nu32 / E3

            C[2, 0] = -nu13 / E1
            C[2, 1] = -nu23 / E2
            C[2, 2] = 1 / E3

            C[3, 3] = 1 / G23
            C[4, 4] = 1 / G13
            C[5, 5] = 1 / G12

            C_laminas.append(np.linalg.inv(C))

        C_laminas = np.array(C_laminas)  # shape = (n_laminas, 6, 6)

        # Preenche C_all usando os índices de cada ponto
        for i_lam, C in enumerate(C_laminas):
            mask = (index == i_lam)
            if np.any(mask):
                # Broadcasting para todos os pontos que pertencem à lâmina
                C_all[:, :, mask] = C[:, :, np.newaxis]

        return C_all


class Lamina:
    def __init__(self, E_11, E_22, E_33, nu_12, nu_13, nu_23, G_12, G_13, G_23, density, angle, thickness):
        self.E_11 = E_11
        self.E_22 = E_22
        self.E_33 = E_33

        self.nu_12 = nu_12
        self.nu_13 = nu_13
        self.nu_23 = nu_23

        self.G_12 = G_12
        self.G_13 = G_13
        self.G_23 = G_23

        self.density = density

        self.angle = angle

        self.thickness = thickness
