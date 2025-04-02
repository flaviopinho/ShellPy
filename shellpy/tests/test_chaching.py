import time
import sympy as sym
import numpy as np

from shellpy.expansions.eigen_function_expansion import EigenFunctionExpansion
from shellpy import RectangularMidSurfaceDomain
from shellpy import xi1_, xi2_, MidSurfaceGeometry

"""
This script is designed to test the performance of certain computational functions by measuring their execution 
time before and after caching their results. The two key functions being tested are natural_base and 
christoffel_symbols_first_derivative from the MidSurfaceGeometry class, which are both computationally expensive.
"""

if __name__ == "__main__":
    # Define parameters for the geometry and material properties
    R = 0.1  # Radius
    a = 0.1  # Length in x direction
    b = 0.1  # Length in y direction
    h = 0.0001  # Thickness of the material

    E = 10  # Young's modulus
    nu = 0.2  # Poisson's ratio

    # Define the edges of the domain as a rectangular region
    edges = RectangularMidSurfaceDomain(0, a, 0, b)

    # Set the expansion sizes for the displacement fields in different directions
    expansion_size = {"u1": (1, 1),
                      "u2": (1, 1),
                      "u3": (1, 1)}

    # Boundary conditions for each displacement field
    boundary_conditions_u1 = {"xi1_0": ("S", "S"),
                              "xi2_0": ("S", "S")}
    boundary_conditions_u2 = {"xi1_0": ("S", "S"),
                              "xi2_0": ("S", "S")}
    boundary_conditions_u3 = {"xi1_0": ("S", "S"),
                              "xi2_0": ("S", "S")}

    # Group mid_surface_domain conditions into a single dictionary
    boundary_conditions = {"u1": boundary_conditions_u1,
                           "u2": boundary_conditions_u2,
                           "u3": boundary_conditions_u3}

    # Create an EigenFunctionExpansion object with the defined parameters
    displacement_field = EigenFunctionExpansion(expansion_size, edges, boundary_conditions)

    # Define the geometry of the mid-surface using sympy for symbolic calculations
    R_ = sym.Matrix([xi1_, xi2_, sym.sqrt(R ** 2 - (xi1_ - a / 2) ** 2 - (xi2_ - b / 2) ** 2)])
    mid_surface_geometry = MidSurfaceGeometry(R_)

    # Create a meshgrid for the xi1_0 and xi2_0 coordinates for numerical evaluation
    xi1 = np.linspace(*edges.edges["xi1"], 500)
    xi2 = np.linspace(*edges.edges["xi2"], 500)
    x, y = np.meshgrid(xi1, xi2, indexing='ij')

    # First calculation: natural_base function without caching
    start = time.time()  # Start the timer
    data = mid_surface_geometry.natural_base(x, y)  # Compute the natural base (e.g., base functions)
    end = time.time()  # End the timer

    # Print the time taken for the first calculation
    print('\n(1) The function natural_base took {:.2f} s to compute.'.format(end - start))
    print(np.shape(data))  # Print the shape of the output

    # Second calculation: natural_base function with potential caching
    start = time.time()  # Start the timer
    data2 = mid_surface_geometry.natural_base(x, y)  # Compute the natural base again (with possible cache)
    end = time.time()  # End the timer

    # Print the time taken for the second calculation
    print('\n(2) The function natural_base took {:.2f} s to compute.'.format(end - start))
    print(np.shape(data2))  # Print the shape of the output

    # Third calculation: natural_base function again (testing multiple calls)
    start = time.time()  # Start the timer
    data2 = mid_surface_geometry.natural_base(x, y)  # Compute the natural base again
    end = time.time()  # End the timer

    # Print the time taken for the third calculation
    print('\n(3) The function natural_base took {:.2f} s to compute.'.format(end - start))
    print(np.shape(data2))  # Print the shape of the output

    # First calculation: christoffel_symbols_first_derivative function without caching
    start = time.time()  # Start the timer
    data = mid_surface_geometry.christoffel_symbols_first_derivative(x, y)  # Compute Christoffel symbols first derivative
    end = time.time()  # End the timer

    # Print the time taken for the first calculation of Christoffel symbols
    print('\n(1) The function christoffel_symbols_first_derivative took {:.2f} s to compute.'.format(end - start))
    print(np.shape(data))  # Print the shape of the output

    # Second calculation: christoffel_symbols_first_derivative function with potential caching
    start = time.time()  # Start the timer
    data2 = mid_surface_geometry.christoffel_symbols_first_derivative(x, y)  # Compute again (with cache)
    end = time.time()  # End the timer

    # Print the time taken for the second calculation
    print('\n(2) The function christoffel_symbols_first_derivative took {:.2f} s to compute.'.format(end - start))
    print(np.shape(data2))  # Print the shape of the output

    # Third calculation: christoffel_symbols_first_derivative function again (testing multiple calls)
    start = time.time()  # Start the timer
    data2 = mid_surface_geometry.christoffel_symbols_first_derivative(x, y)  # Compute again
    end = time.time()  # End the timer

    # Print the time taken for the third calculation
    print('\n(3) The function christoffel_symbols_first_derivative took {:.2f} s to compute.'.format(end - start))
    print(np.shape(data2))  # Print the shape of the output
