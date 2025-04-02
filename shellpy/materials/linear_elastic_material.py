import numpy as np

class LinearElasticMaterial:
    """
    A class to represent the material properties of a linear elastic material.
    It includes methods to compute the constitutive tensor for thin shells
    based on the material's properties: Young's modulus (E), Poisson's ratio (nu),
    and material density.
    """

    def __init__(self, E, nu, density):
        """
        Initialize the material properties.

        :param E: Young's modulus of the material (in Pascals).
        :param nu: Poisson's ratio of the material (dimensionless).
        :param density: Density of the material (in kg/m^3).
        """
        self.E = E  # Young's modulus (elasticity)
        self.nu = nu  # Poisson's ratio (relates lateral strain to axial strain)
        self.density = density  # Density of the material