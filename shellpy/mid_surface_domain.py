from abc import ABC, abstractmethod


class MidSurfaceDomain(ABC):
    """
    Abstract base class for defining a mid-surface domain.
    Any class representing a mid-surface domain should inherit from this class
    and implement the necessary methods and properties.
    """
    pass


class RectangularMidSurfaceDomain(MidSurfaceDomain):
    """
    Represents a rectangular mid-surface domain defined by the edges of two curvilinear coordinates, xi1 and xi2.
    The edges are specified by the values of xi1 and xi2 at their respective boundaries.
    """

    def __init__(self, xi1_a, xi1_b, xi2_a, xi2_b):
        """
        Initializes the rectangular mid-surface domain with the given edge values.

        :param xi1_a: The lower mid_surface_domain value of the first curvilinear coordinate (xi1).
        :param xi1_b: The upper mid_surface_domain value of the first curvilinear coordinate (xi1).
        :param xi2_a: The lower mid_surface_domain value of the second curvilinear coordinate (xi2).
        :param xi2_b: The upper mid_surface_domain value of the second curvilinear coordinate (xi2).

        This sets up the 'edges' dictionary which stores the boundaries of xi1 and xi2.
        """
        # Store the edge values for xi1 and xi2 in a dictionary for easy access
        self.edges = {"xi1": (xi1_a, xi1_b),
                      "xi2": (xi2_a, xi2_b)}

    @property
    def xi1_a(self):
        """
        Returns the lower mid_surface_domain of the first curvilinear coordinate (xi1).
        """
        return self.edges["xi1"][0]

    @property
    def xi1_b(self):
        """
        Returns the upper mid_surface_domain of the first curvilinear coordinate (xi1).
        """
        return self.edges["xi1"][1]

    @property
    def xi2_a(self):
        """
        Returns the lower mid_surface_domain of the second curvilinear coordinate (xi2).
        This is represented as a function of xi1 since it is constant with respect to xi1.
        """
        return lambda xi1: self.edges["xi2"][0]

    @property
    def xi2_b(self):
        """
        Returns the upper mid_surface_domain of the second curvilinear coordinate (xi2).
        This is represented as a function of xi1 since it is constant with respect to xi1.
        """
        return lambda xi1: self.edges["xi2"][1]
