from shellpy import pinned
from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy import RectangularMidSurfaceDomain
import sympy as sym
from shellpy.koiter_shell_theory import fast_koiter_quadratic_strain_energy, fast_koiter_strain_energy

from shellpy import LinearElasticMaterial
from shellpy import MidSurfaceGeometry, xi1_, xi2_
from shellpy import Shell
from shellpy import ConstantThickness

if __name__ == "__main__":

    # Define the shell geometry parameters.
    R = 1  # Radius of the spherical shell (in meters).
    a = 0.1  # Length in the xi1_0 direction (length of the rectangular domain in meters).
    b = 0.1  # Length in the xi2_0 direction (width of the rectangular domain in meters).
    h = 0.001  # Thickness of the shell (in meters).

    # Define the material properties for the shell.
    E = 206E9  # Young's modulus in Pascals (for steel).
    nu = 0.3  # Poisson's ratio (typical for steel).
    density = 7850  # Density of the material in kg/m^3 (for steel).

    # Define the mid-surface geometry of the shell as a rectangular domain.
    rectangular_domain = RectangularMidSurfaceDomain(0, a, 0, b)

    # Define the displacement field expansion size for the shell.
    # This defines the number of terms in the displacement series for each direction.
    expansion_size = {"u1": (1, 1),  # Number of terms in the displacement series in the xi1_0 direction.
                      "u2": (1, 1),  # Number of terms in the displacement series in the xi2_0 direction.
                      "u3": (1, 1)}  # Number of terms in the displacement series in the xi3 direction.

    # Define rectangular_domain conditions for the shell.
    boundary_conditions = pinned  # This assumes a pinned rectangular_domain condition (edges are fixed).

    # Define the displacement field using trigonometric series for the shell.
    # This defines the shape functions for the displacement, with the given expansion size and rectangular_domain conditions.
    displacement_field = EigenFunctionExpansion(expansion_size, rectangular_domain, boundary_conditions)

    # Define the symbolic representation for the mid-surface geometry.
    # The coordinates are defined using symbolic variables xi1_0 and xi2_0, with xi3 depending on the radius.
    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])

    # Create the MidSurfaceGeometry object, which models the geometry of the shell.
    mid_surface_geometry = MidSurfaceGeometry(R_)

    # Create the thickness function, which provides the shell thickness.
    thickness = ConstantThickness(h)

    # Create the material object that encapsulates the material properties (Elastic Modulus, Poisson’s Ratio, and Density).
    material = LinearElasticMaterial(E, nu, density)

    # Create the Shell object, which combines all the above components (geometry, material, displacement field).
    shell = Shell(mid_surface_geometry, thickness, rectangular_domain, material, displacement_field, None)

    # Compute the strain energy using the Koiter shell theory functions.
    # This function calculates the quadratic, cubic and quartic strain energy for the shell under the defined configuration.
    U1, U2, U3 = fast_koiter_strain_energy(shell)



