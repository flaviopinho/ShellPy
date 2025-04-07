class FunctionallyGradedMaterial:
    def __init__(self, E_0, E_1, nu_0, nu_1, density_0, density_1, Vc):
        self.E_0 = E_0
        self.E_1 = E_1
        self.nu_0 = nu_0
        self.nu_1 = nu_1
        self.density_0 = density_0
        self.density_1 = density_1
        self.Vc = Vc

    def E(self, z):
        return (self.E_0 - self.E_1) * self.Vc(z) + self.E_1

    def nu(self, z):
        return (self.nu_0 - self.nu_1) * self.Vc(z) + self.nu_1

    def density(self, z):
        return (self.density_0 - self.density_1) * self.Vc(z) + self.density_1


def power_law_distribution1(z, h, a, b, c, p):
    return (1 - a * (1 / 2 + z / h) + b * (1 / 2 + z / h) ** c) ** p


def power_law_distribution2(z, h, a, b, c, p):
    return (1 - a * (1 / 2 - z / h) + b * (1 / 2 - z / h) ** c) ** p
