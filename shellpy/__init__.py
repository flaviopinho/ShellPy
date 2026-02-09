"""
ShellPy: Python library for shell analysis using Ritz-based formulations
and advanced shell theories.
"""

__version__ = "0.2.0"

# Core geometry / domain
from .mid_surface_domain import RectangularMidSurfaceDomain
from .midsurface_geometry import MidSurfaceGeometry, xi1_, xi2_
from .shell import Shell
from .displacement_expansion import DisplacementExpansion
from .thickness import ConstantThickness
from .continuationpy import Continuation
from .cache_decorator import cache_function
from .cache_decorator import cache_method
from .cache_decorator import cache_global


__all__ = [
    "Shell",
    "RectangularMidSurfaceDomain",
    "MidSurfaceGeometry",
    "ConstantThickness",
    "xi1_",
    "xi2_",
    "DisplacementExpansion",
    "Continuation",
    "cache_method",
    "cache_function",
    "cache_global",
]