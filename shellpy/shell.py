from typing import Callable, Any

from .displacement_expansion import DisplacementExpansion
from .mid_surface_domain import MidSurfaceDomain
from .midsurface_geometry import MidSurfaceGeometry


class Shell:
    """
    This class represents a shell structure with its geometry, material properties, displacement field, mid_surface_domain conditions, and applied load.
    """

    def __init__(self,
                 mid_surface_geometry: MidSurfaceGeometry,  # Mid-surface geometry of the shell (e.g., shape, curvature)
                 thickness: Callable[[Any, Any], Any],  # Thickness function of the shell, which may depend on position (xi1, xi2)
                 mid_surface_domain: MidSurfaceDomain,  # Mid Surface curvilinear coordinates domain
                 material,  # Material properties (e.g., Young's modulus, Poisson's ratio, density)
                 displacement_field: DisplacementExpansion,  # Displacement field expansion, which defines the displacement in terms of shape functions
                 load):  # Applied load function, which may depend on position and time
        """
        Initializes the properties of the shell structure.

        :param mid_surface_geometry: The geometry of the shell's mid-surface (e.g., curvature, shape).
        :param thickness: A function that returns the thickness of the shell at any given point (xi1, xi2).
        :param mid_surface_domain: Mid-Surface curvilinear coordinates' domain.
        :param material: The material properties of the shell, such as Young's modulus, Poisson's ratio, and density.
        :param displacement_field: The displacement field expansion, defining the displacement of the shell using shape functions.
        :param load: The applied load, which can be a function depending on position and time.
        """
        # Assign the provided parameters to the instance variables
        self.mid_surface_geometry = mid_surface_geometry  # Geometry of the shell's mid-surface
        self.thickness = thickness  # Function defining the shell's thickness at any given point
        self.mid_surface_domain = mid_surface_domain  # Mid-Surface curvilinear coordinates' domain.
        self.material = material  # Material properties (e.g., Young's modulus, Poisson's ratio)
        self.displacement_expansion = displacement_field  # Displacement expansion field, describing the displacement in terms of shape functions
        self.load = load  # Applied load, which could vary over time or position
