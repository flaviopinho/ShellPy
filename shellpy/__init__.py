"""
ShellPy: Python library for shell analysis using Ritz-based formulations
and advanced shell theories.
"""

# Core geometry / domain
from .mid_surface_domain import RectangularMidSurfaceDomain
from .midsurface_geometry import MidSurfaceGeometry, xi1_, xi2_
from .shell import Shell
from .displacement_expansion import DisplacementExpansion
from .thickness import ConstantThickness
from .tensor_derivatives import tensor_derivative
from .continuationpy import Continuation


__all__ = [
    "Shell",
    "RectangularMidSurfaceDomain",
    "MidSurfaceGeometry",
    "ConstantThickness",
    "xi1_",
    "xi2_",
    "DisplacementExpansion",
    "tensor_derivative"
    "Continuation"
]