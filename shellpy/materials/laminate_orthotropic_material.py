import numpy as np


class LaminateOrthotropicMaterial:
    def __init__(self, laminas):
        """
        laminas: lista de objetos Lamina (de baixo para cima)
        """
        self.laminas = laminas
        self.total_thickness = sum(l.thickness for l in laminas)

        if not np.isclose(self.total_thickness, 2.0, atol=1e-8):
            raise ValueError(
                f"Thickness must be 2, but {self.total_thickness}"
            )

        self.mass_per_area = self._compute_mass()

    def _compute_mass(self):
        """Massa por unidade de área do laminado."""
        return sum(l.density * l.thickness for l in self.laminas)

    def angle(self, xi3_norm):
        """
        Retorna o ângulo da lâmina em que xi3 está localizado.
        Se xi3 for um array, retorna um array de ângulos.
        """
        xi3_norm = np.atleast_1d(xi3_norm)  # garante array
        z_bot = -self.total_thickness / 2

        # limites acumulados das lâminas
        z_tops = []
        z = z_bot
        for l in self.laminas:
            z += l.thickness
            z_tops.append(z)

        alphas = []
        for val in xi3_norm:
            for l, z_top in zip(self.laminas, z_tops):
                if val <= z_top:
                    alphas.append(l.angle)
                    break
            else:
                raise ValueError(f"xi3={val} fora do intervalo do laminado.")

        alphas = np.array(alphas)
        return alphas.reshape(xi3_norm.shape)  # mesma shape de entrada

    def orthotropic_voigt_matrix(self, xi3_norm):
        """
        Retorna matriz constitutiva (6x6) para cada ponto xi3_norm.
        Output: ndarray com shape (6,6) + xi3_norm.shape
        """
        xi3_norm = np.atleast_1d(xi3_norm)  # garante array
        z_bot = -self.total_thickness / 2

        # limites acumulados das lâminas
        z_tops = []
        z = z_bot
        for l in self.laminas:
            z += l.thickness
            z_tops.append(z)

        C_all = []
        for xi3 in xi3_norm:
            # encontra a lâmina
            for l, z_top in zip(self.laminas, z_tops):
                if xi3 <= z_top:
                    lamina = l
                    break
            else:
                raise ValueError(f"xi3={xi3} fora do intervalo do laminado.")

            # extrai propriedades
            E1, E2, E3 = lamina.E_11, lamina.E_22, lamina.E_33
            nu12, nu21 = lamina.nu_12, lamina.nu_21
            nu13, nu31 = lamina.nu_13, lamina.nu_31
            nu23, nu32 = lamina.nu_23, lamina.nu_32

            # aqui você pode definir G12,G13,G23 (depende se já tem ou precisa calcular)
            # por enquanto coloco placeholders:
            G12 = E1 / (2 * (1 + nu12))
            G13 = E1 / (2 * (1 + nu13))
            G23 = E2 / (2 * (1 + nu23))

            # fator determinante
            Delta = (1 - nu12 * nu21 - nu23 * nu32 - nu13 * nu31
                     - 2 * nu12 * nu23 * nu31)

            # monta matriz constitutiva
            C = np.zeros((6, 6))

            C[0, 0] = E1 * (1 - nu23 * nu32) / Delta
            C[0, 1] = E1 * (nu21 + nu23 * nu31) / Delta
            C[0, 2] = E1 * (nu31 + nu21 * nu32) / Delta

            C[1, 0] = C[0, 1]
            C[1, 1] = E2 * (1 - nu13 * nu31) / Delta
            C[1, 2] = E2 * (nu32 + nu31 * nu12) / Delta

            C[2, 0] = C[0, 2]
            C[2, 1] = C[1, 2]
            C[2, 2] = E3 * (1 - nu12 * nu21) / Delta

            C[3, 3] = G23
            C[4, 4] = G13
            C[5, 5] = G12

            C_all.append(C)

        # empilha e reordena no formato desejado
        C_all = np.stack(C_all, axis=-1)  # shape (6,6,N)
        return C_all.reshape((6, 6) + xi3_norm.shape)


class Lamina:
    def __init__(self, E_11, E_22, E_33, nu_12, nu_21, nu_13, nu_31, nu_23, nu_32, density, theta, thickness):
        self.E_11 = E_11
        self.E_22 = E_22
        self.E_33 = E_33

        self.nu_12 = nu_12
        self.nu_21 = nu_21
        self.nu_13 = nu_13
        self.nu_31 = nu_31
        self.nu_23 = nu_23
        self.nu_32 = nu_32

        self.density = density

        self.angle = theta

        self.thickness = thickness
