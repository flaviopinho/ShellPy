import numpy as np

from .cache_decorator import cache_method


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
        self.cache = {}  # A cache to store precomputed results for efficiency

    @cache_method  # This decorator caches the results of the method to avoid redundant computations.
    def thin_shell_constitutive_tensor(self, metric_tensor=None):
        """
        Computes the constitutive tensor for a thin shell based on the given metric tensor.

        :param metric_tensor: The metric tensor used to compute the constitutive tensor (default is identity).
        :return: The constitutive tensor of the thin shell, computed using the material properties and metric tensor.
        """
        # If no metric tensor is provided, initialize it as a 2x2 identity tensor for the default case.
        if metric_tensor is None:
            # Creating a 4-dimensional tensor (2, 2, 2, 2) with identity in the diagonal.
            metric_tensor = np.zeros((2,) * 4)
            metric_tensor[tuple([np.arange(2)] * 4)] = 1  # Identity tensor (diagonal elements set to 1)

        # Calculate the first term of the constitutive tensor (T1)
        # This is a contraction of the metric tensor with itself.
        T1 = np.einsum('ij...,kl...->iklj...', metric_tensor, metric_tensor)

        # Calculate the second term of the constitutive tensor (T2)
        # This is a contraction of the metric tensor with itself, but with different indices.
        T2 = np.einsum('ij...,kl...->ikjl...', metric_tensor, metric_tensor)

        # Calculate the third term of the constitutive tensor (T3)
        # This is a contraction of the metric tensor with itself, with all indices intact.
        T3 = np.einsum('ij...,kl...->ijkl...', metric_tensor, metric_tensor)

        # Compute the constitutive tensor using the material properties E (Young's modulus) and nu (Poisson's ratio)
        # The formula incorporates the material's behavior and the metric tensor's influence.
        return self.E / (2 * (1 - self.nu ** 2)) * ((1 - self.nu) * (T1 + T2) + 2 * self.nu * T3)
