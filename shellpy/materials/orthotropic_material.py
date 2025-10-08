import numpy as np


class OrthotropicMaterial:
    def __init__(self, E_11, E_22, E_33, nu_12, nu_13, nu_23, G_12, G_13, G_23, density, material_base=None):
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
        if material_base is None:
            e1_material = np.array((1, 0, 0))
            e2_material = np.array((0, 1, 0))
            e3_material = np.array((0, 0, 1))
            self.material_base = np.stack((e1_material, e2_material, e3_material), axis=0)
        else:
            self.material_base = material_base

    def orthotropic_voigt_matrix(self):
        """
        Retorna a matriz constitutiva ortotrópica (6x6) para cada ponto xi3.
        Output: ndarray com shape (6,6) + xi3.shape
        """

        E1, E2, E3 = self.E_11, self.E_22, self.E_33
        nu12 = self.nu_12
        nu13 = self.nu_13
        nu23 = self.nu_23

        nu21 = nu12 * E2 / E1
        nu31 = nu13 * E3 / E1
        nu32 = nu23 * E3 / E2

        G12 = E1 / (2 * (1 + nu12))
        G13 = E1 / (2 * (1 + nu13))
        G23 = E2 / (2 * (1 + nu23))

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

        C[3, 3] = 1 / G23 * (6 / 5)
        C[4, 4] = 1 / G13 * (6 / 5)
        C[5, 5] = 1 / G12

        return np.linalg.inv(C)
