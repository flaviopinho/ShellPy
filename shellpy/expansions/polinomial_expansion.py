import numpy as np
from numpy import polynomial as P

from ..displacement_expansion import DisplacementExpansion, displacement_field_index
from ..mid_surface_domain import RectangularMidSurfaceDomain

from .simple_expansions import fourier_expansion_for_periodic_solutions


class GenericPolynomialSeries(DisplacementExpansion):
    """
    Defines a displacement expansion using polynomials for standard boundaries (S, F, C)
    and Fourier series for periodic boundaries (R).
    """

    def __init__(self, function, expansion_size, rectangular_domain: RectangularMidSurfaceDomain, boundary_conditions,
                 mapping=None):
        """
        Create a displacement expansion
        :param function: np.polynomial.Legendre or np.polynomial.Chebyshev
        :param expansion_size: Dict of tuple.
                               example: expansion_size = {"u1": (5, 5), "u2": (5, 5), "u3": (5, 5)}
        :param rectangular_domain: RectangularMidSurfaceDomain object representing the domain.
        :param boundary_conditions: Dict of dicts of tuples
                                    example:
                                    boundary_conditions = {"u1": {"xi1": ("S", "S"), "xi2": ("R", "R")},
                                                           "u2": {"xi1": ("S", "S"), "xi2": ("R", "R")}, ...}
        :param mapping: Optional. List of tuples specifying which modes are included.
        """
        self.function = function
        self._expansion_size = expansion_size
        self._edges = rectangular_domain.edges
        self._boundary_conditions = boundary_conditions

        if mapping is None:
            self._mapping = self._set_mapping()
        else:
            self._mapping = mapping

        self._number_of_fields = len(expansion_size)

        # Instead of storing _bc_equations and _coeff as before,
        # we generate a set of callable equations, similar to EnrichedCosineExpansion.
        self._equations = self._determine_equations()

    def _set_mapping(self):
        """
        Generates the default mapping based on the expansion size.
        """
        mapping = []
        self._dof = 0
        for key, value in self._expansion_size.items():
            displacement_field = key
            m = value[0]  # Modes in xi1 direction
            n = value[1]  # Modes in xi2 direction
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    mapping.append((displacement_field, i, j))
                    self._dof += 1
        return mapping

    def _determine_equations(self):
        """
        Determines the 1D equations for each field and direction, then combines them
        into a 2D lambda function for shape function evaluations.
        """
        # This dictionary stores the 1D functions for each direction
        functions_1d = {}

        for field, bc_dict in self._boundary_conditions.items():
            # Retrieve the maximum number of modes for this displacement field
            max_modes = [t for t in self._mapping if t[0] == field]
            max_mode_xi1 = max(t[1] for t in max_modes) if max_modes else 0
            max_mode_xi2 = max(t[2] for t in max_modes) if max_modes else 0

            for direction, bc_tuple in bc_dict.items():
                max_mode = max_mode_xi1 if direction == "xi1" else max_mode_xi2
                edges = self._edges[direction]

                # If the boundary condition is R-R (periodic), use the Fourier expansion
                if bc_tuple == ("R", "R"):
                    # The Fourier function returns a dict: {(mode, derivative): lambda x: ...}
                    functions_1d[(field, direction)] = fourier_expansion_for_periodic_solutions(
                        edges, 3, max_mode + 1
                    )
                else:
                    # For non-periodic boundaries, use the Chebyshev/Legendre polynomial logic
                    functions_1d[(field, direction)] = self._generate_polynomial_set(
                        edges, bc_tuple, max_mode
                    )

        # Create a 2D lambda function that combines both directions
        equations = lambda xi1, xi2, field, i, j, deriv1, deriv2: (
                functions_1d[(field, "xi1")][(i - 1, deriv1)](xi1) *
                functions_1d[(field, "xi2")][(j - 1, deriv2)](xi2)
        )

        return equations

    def _generate_polynomial_set(self, edges, boundary_conditions, maximum_mode, maximum_derivative=3):
        """
        Generates 1D functions using Legendre/Chebyshev polynomials for a specific direction.
        :param edges: Tuple representing the start and end of the domain.
        :param boundary_conditions: Tuple representing left and right boundary conditions (e.g., ("S", "F")).
        :param maximum_mode: Integer for the maximum expansion mode.
        :param maximum_derivative: Integer for the required highest derivative.
        :return: A dictionary in the format {(mode, derivative): lambda x: value}
        """
        left_bc, right_bc = boundary_conditions

        # 1. Determine the boundary condition polynomial B(x)
        def get_bc_poly(edge_val, bc_type):
            if bc_type in ("F", "O"):
                return P.Polynomial([1])
            elif bc_type == "S":
                return P.Polynomial([edge_val, -1])
            elif bc_type == "C":
                return P.Polynomial([edge_val, -1]) ** 2
            else:
                raise ValueError(f"Invalid boundary condition type: {bc_type}")

        poly_bc = get_bc_poly(edges[0], left_bc) * get_bc_poly(edges[1], right_bc)

        functions_set = {}

        # 2. Generate the basis functions P_i(x) multiplied by B(x)
        for i in range(1, maximum_mode + 1):
            coeff = [0] * i
            coeff[i - 1] = 1

            # Generate the basis (Legendre or Chebyshev) and convert it to a standard polynomial
            base_poly = self.function(coeff, domain=list(edges)).convert(kind=P.Polynomial)

            # U_i(x) = B(x) * P_i(x)
            u_poly = poly_bc * base_poly

            # Store the derivatives as lambda functions for fast evaluation
            for deriv in range(maximum_derivative):
                u_deriv = u_poly.deriv(deriv)

                # CAUTION: Local scope must be forced in the Python lambda (using default arguments)
                functions_set[(i - 1, deriv)] = lambda x, p=u_deriv: p(x)

        return functions_set

    def shape_function(self, n, xi1, xi2, derivative1=0, derivative2=0):
        """
        Determine the shape function and its derivatives for the coordinates xi1 and xi2
        """
        u = np.zeros((self._number_of_fields,) + np.shape(xi1), dtype=np.float64)
        field = self._mapping[n][0]
        k = displacement_field_index[field]
        i = self._mapping[n][1]
        j = self._mapping[n][2]

        c = self._equations(xi1, xi2, field, i, j, derivative1, derivative2)
        u[k] = c
        return u

    def shape_function_first_derivatives(self, n, xi1, xi2):
        """
        Determine the first derivative of the shape function with respect to xi1 and xi2
        """
        du = np.zeros((self._number_of_fields, 2) + np.shape(xi1), dtype=np.float64)
        field = self._mapping[n][0]
        k = displacement_field_index[field]
        i = self._mapping[n][1]
        j = self._mapping[n][2]

        for derivative_index, derivatives in enumerate(((1, 0), (0, 1))):
            du[k, derivative_index] = self._equations(xi1, xi2, field, i, j, *derivatives)
        return du

    def shape_function_second_derivatives(self, n, xi1, xi2):
        """
        Determine the second derivative of the shape function with respect to xi1 and xi2
        """
        ddu = np.zeros((self._number_of_fields, 2, 2) + np.shape(xi1), dtype=np.float64)
        field = self._mapping[n][0]
        k = displacement_field_index[field]
        i = self._mapping[n][1]
        j = self._mapping[n][2]

        for jj, derivatives_j in enumerate(((1, 0), (0, 1))):
            for kk, derivatives_k in enumerate(((1, 0), (0, 1))):
                derivatives = (derivatives_j[0] + derivatives_k[0], derivatives_j[1] + derivatives_k[1])
                ddu[k, jj, kk] = self._equations(xi1, xi2, field, i, j, *derivatives)
        return ddu

    def mapping(self, n):
        """
        Return the corresponding field and modes for the n-th degree of freedom.
        """
        return self._mapping[n]

    def number_of_degrees_of_freedom(self):
        """
        Return the total number of degrees of freedom.
        """
        return len(self._mapping)

    def number_of_fields(self):
        """
        Return the number of displacement fields.
        """
        return self._number_of_fields

    def __call__(self, *args, **kwargs):
        """
        Evaluate the complete displacement field.
        """
        U = args[0]
        xi1 = args[1]
        xi2 = args[2]
        result = np.zeros((self.number_of_fields(),) + np.shape(xi1))
        for i in range(self.number_of_degrees_of_freedom()):
            result += self.shape_function(i, xi1, xi2, 0, 0) * U[i]
        return result


LegendreSeries = lambda expansion_size, rectangular_boundary, boundary_conditions, mapping=None: GenericPolynomialSeries(P.Legendre,
                                                              expansion_size,
                                                              rectangular_boundary,
                                                              boundary_conditions,
                                                              mapping)

ChebyshevSeries = lambda expansion_size, rectangular_boundary, boundary_conditions, mapping=None: GenericPolynomialSeries(P.Chebyshev,
                                                               expansion_size,
                                                               rectangular_boundary,
                                                               boundary_conditions,
                                                               mapping)
