import numpy as np
from numpy import polynomial as P

from shellpy import DisplacementExpansion, displacement_field_index, cache_method
from shellpy import RectangularMidSurfaceDomain


class GenericPolynomialSeries(DisplacementExpansion):
    """
    Define a displacement expansion using polynomials
    """

    def __init__(self, function, expansion_size, rectangular_domain: RectangularMidSurfaceDomain, boundary_conditions,
                 mapping=None):
        """
        Create a displacement expansion
        :param function: np.polynomial.Legendre or np.polynomial.Chebyshev
        :param expansion_size: Dict of tuple.
                               exemple: expansion_size = {"u1": (5, 5),
                                                          "u2": (5, 5),
                                                          "u3": (5, 5)}
        :param rectangular_domain: Dict of tuples.
                        exemple:
                        edges = {"xi1": (0, a),
                                 "xi2": (0, b)}
        :param boundary_conditions: Dict of dicts of tuples
                                    exemple:
                                    boundary_conditions = {"u1": {"xi1": ("S", "S"),
                                                                  "xi2": ("S", "S")},
                                                           "u2": {"xi1": ("S", "S"),
                                                                  "xi2": ("S", "S")},
                                                           "u3": {"xi1": ("S", "S"),
                                                                  "xi2": ("S", "S")}

        :param mapping: List of tuples. Exemple: [('u1', 1, 1), ('u2, 1, 2), ...]
                        Specify which modes are included in the expansion.
                        Must be compatible with expansion size
        """
        self.function = function
        self._expansion_size = expansion_size
        if mapping is None:
            self._mapping = self._set_mapping()
        else:
            self._mapping = mapping
        self._edges = rectangular_domain.edges
        self._boundary_conditions = boundary_conditions
        self._bc_equations = self._set_boundary_conditions_equations()
        self._coeff = self._determine_coefficients()

        self._number_of_fields = len(expansion_size)
        if self._number_of_fields not in (3, 6):
            ValueError('Expansion must have 3 or 6 fields.')

        self.cache = {}

    def _set_mapping(self):
        mapping = []
        self._dof = 0
        for key, value in self._expansion_size.items():
            displacement_field = key
            m = value[0]
            n = value[1]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    _i = i
                    _j = j
                    mapping.append((displacement_field, _i, _j))
                    self._dof += 1

        return mapping

    def _set_boundary_conditions_equations(self):
        bc_equations = {}
        for displacement, bc_type in self._boundary_conditions.items():
            for direction in ("xi1", "xi2"):
                left_type, right_type = bc_type[direction]
                edges = self._edges[direction]
                poly1 = self._get_boundary_condition_equation(edges[0], left_type)
                poly2 = self._get_boundary_condition_equation(edges[1], right_type)
                bc_equations[(displacement, direction)] = poly1 * poly2
        return bc_equations

    def _get_boundary_condition_equation(self, edge, condition_type) -> P.Polynomial:
        if condition_type == "F" or condition_type == "O":
            return P.Polynomial([1])
        elif condition_type == "S":
            return P.Polynomial([edge, -1])
        elif condition_type == "C":
            return P.Polynomial([edge, -1]) * P.Polynomial([edge, -1])
        else:
            raise ValueError(f"Invalid boundary condition type: {condition_type}")

    def _function_field(self, i, j):
        coeff1 = [0] * i
        coeff1[i - 1] = 1
        coeff2 = [0] * j
        coeff2[j - 1] = 1
        u_x = self.function(coeff1, domain=list(self._edges["xi1"])).convert(kind=P.Polynomial)
        u_y = self.function(coeff2, domain=list(self._edges["xi2"])).convert(kind=P.Polynomial)

        return u_x, u_y

    def _determine_coefficients(self):
        coeff = {}
        for k, (displacement_field, i, j) in enumerate(self._mapping):
            leg_dir1, leg_dir2 = self._function_field(i, j)
            bc_dir1 = self._bc_equations[(displacement_field, 'xi1')]
            bc_dir2 = self._bc_equations[(displacement_field, 'xi2')]
            u_dir1 = bc_dir1 * leg_dir1
            u_dir2 = bc_dir2 * leg_dir2

            for derivative1 in range(3):
                coef1 = u_dir1.deriv(derivative1).coef
                for derivative2 in range(3):
                    coef2 = u_dir2.deriv(derivative2).coef
                    coeff[(k, derivative1, derivative2)] = np.outer(coef1, coef2)

        return coeff

    def mapping(self, n):
        """
        :param n: n-th degree of freedom
        :return: A list containing the correspondent field and mode1 and mode2 of the n-th dof
        """
        return self._mapping[n]

    def number_of_degrees_of_freedom(self):
        return len(self._mapping)

    def shape_function(self, n, xi1, xi2, derivative1=0, derivative2=0):
        """
        Determine the shape function and its derivatives for the coordinates xi1 and xi2
        :param n: index of the n-th shape function
        :param xi1: curvilinear coordinate 1
        :param xi2: curvilinear coordinate 2
        :param derivative1: Optional. Positive integer that represents the derivative of xi1
        :param derivative2: Optional. Positive integer that represents the derivative of xi2
        :return: A vector that contains the value of the shape function for the coordinates xi1 and xi2
        """
        c = self._coeff[(n, derivative1, derivative2)]
        value = P.polynomial.polyval2d(xi1, xi2, c)
        vector_u = np.zeros((self._number_of_fields,) + np.shape(xi1))
        vector_u[displacement_field_index[self._mapping[n][0]]] = value
        return vector_u

    def shape_function_first_derivatives(self, n, xi1, xi2):
        """
        Determine the first derivative of the shape function with respect to xi1 and xi2
        :param n: index of the n-th shape function
        :param xi1: curvilinear coordinate 1
        :param xi2: curvilinear coordinate 2
        :return: Returns a matrix (numpy ndarray) that contains the derivative of the shape_functions
        """
        du = np.zeros((self._number_of_fields, 2) + np.shape(xi1), dtype=np.float64)
        i = displacement_field_index[self._mapping[n][0]]
        for jj, j in enumerate(((1, 0), (0, 1))):
            c = self._coeff[(n,) + j]
            du[i, jj] = P.polynomial.polyval2d(xi1, xi2, c)

        return du

    def shape_function_second_derivatives(self, n, xi1, xi2):
        """
        Determine the second derivative of the shape function with respect to xi1 and xi2
        :param n: index of the n-th shape function
        :param xi1: curvilinear coordinate 1
        :param xi2: curvilinear coordinate 2
        :return: Returns a tensor (numpy ndarray) that contains the derivative of the shape_functions
        """
        ddu = np.zeros((self._number_of_fields, 2, 2) + np.shape(xi1), dtype=np.float64)
        i = displacement_field_index[self._mapping[n][0]]

        for jj, j in enumerate(((1, 0), (0, 1))):
            for kk, k in enumerate(((1, 0), (0, 1))):
                aux = (n, j[0] + k[0], j[1] + k[1])
                c = self._coeff[aux]
                ddu[i, jj, kk] = P.polynomial.polyval2d(xi1, xi2, c)

        return ddu

    def __call__(self, *args, **kwargs):
        U = args[0]
        xi1 = args[1]
        xi2 = args[2]

        result = np.zeros((self.number_of_fields(),) + np.shape(xi1))

        for i in range(self.number_of_degrees_of_freedom()):
            result += self.shape_function(i, xi1, xi2, 0, 0) * U[i]

        return result

    def number_of_fields(self):
        return self._number_of_fields


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