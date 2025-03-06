import numpy as np

from shellpy import DisplacementExpansion, displacement_field_index
from shellpy import RectangularMidSurfaceDomain
from shellpy.expansions import determine_eigenfunctions
from shellpy.expansions.simple_expansions import constant_value_expansion, fourier_expansion_for_periodic_solutions


class EigenFunctionExpansion(DisplacementExpansion):
    """
    Define a displacement expansion using trigonometric functions obtained from an eigen problem
    """

    def __init__(self, expansion_size, rectangular_domain: RectangularMidSurfaceDomain, boundary_conditions,
                 mapping=None):
        """
        Create a displacement expansion
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

        self._expansion_size = expansion_size
        if mapping is None:
            self._mapping = self._set_mapping()
        else:
            self._mapping = mapping
        self._edges = rectangular_domain.edges
        self._boundary_conditions = boundary_conditions
        self._equations = self._determine_equations()

        self._number_of_fields = len(expansion_size)
        if self._number_of_fields != 3 or self._number_of_fields != 6:
            ValueError('Expansion must have 3 or 6 fields.')

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

    def _determine_equations(self):
        l1 = self._edges["xi1"][1] - self._edges["xi1"][0]
        l2 = self._edges["xi2"][1] - self._edges["xi2"][0]

        b1 = self._edges["xi1"][0]
        b2 = self._edges["xi2"][0]

        eigen_functions = {}
        for field_, bc in self._boundary_conditions.items():

            maximum_modes = [t for t in self._mapping if t[0] == field_]
            if not maximum_modes:
                maximum_modes_xi1 = 0
                maximum_modes_xi2 = 0
            else:
                maximum_modes_xi1 = max(t[1] for t in maximum_modes)
                maximum_modes_xi2 = max(t[2] for t in maximum_modes)

            for direction, BC in bc.items():
                if direction == "xi1":
                    maximum_mode = maximum_modes_xi1
                else:
                    maximum_mode = maximum_modes_xi2

                if BC == ("O", "O"):
                    if maximum_mode != 1:
                        raise ValueError("expansion_size must be 1 when BC is ('O', 'O')")
                    eigen_functions[(field_, direction)] = constant_value_expansion(3)

                elif BC == ("R", "R"):
                    eigen_functions[(field_, direction)] = fourier_expansion_for_periodic_solutions(
                        (0, 1), 3, maximum_mode + 1)
                else:
                    eigen_functions[(field_, direction)] = determine_eigenfunctions(BC, 3, maximum_mode + 3)

        equations = lambda xi1, xi2, field, i, j, derivative1, derivative2: (
                eigen_functions[field, "xi1"][(i - 1, derivative1)]((xi1 - b1) / l1)
                * eigen_functions[field, "xi2"][(j - 1, derivative2)]((xi2 - b2) / l2)
                * (1 / l1 ** derivative1) * (1 / l2 ** derivative2))
        return equations

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

        u = np.zeros((self._number_of_fields,) + np.shape(xi1), dtype=np.float64)
        vec = self._mapping[n]
        field = self._mapping[n][0]
        k = displacement_field_index[field]
        i = vec[1]
        j = vec[2]
        c = self._equations(xi1, xi2, field, i, j, derivative1, derivative2)
        u[k] = c
        if self.number_of_fields() == 6:
            v = u[3:6]
            u = u[0:3]
            return u, v
        else:
            return u

    def shape_function_first_derivatives(self, n, xi1, xi2):
        """
        Determine the first derivative of the shape function with respect to xi1 and xi2
        :param n: index of the n-th shape function
        :param xi1: curvilinear coordinate 1
        :param xi2: curvilinear coordinate 2
        :return: Returns a matrix (numpy ndarray) that contains the derivative of the shape_functions
        """
        du = np.zeros((self._number_of_fields, 2) + np.shape(xi1), dtype=np.float64)
        field = self._mapping[n][0]
        k = displacement_field_index[field]
        i = self._mapping[n][1]
        j = self._mapping[n][2]

        for derivative_index, derivatives in enumerate(((1, 0), (0, 1))):
            du[k, derivative_index] = self._equations(xi1, xi2, field, i, j, *derivatives)

        if self.number_of_fields() == 6:
            dv = du[3:6]
            du = du[0:3]
            return du, dv
        else:
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
        field = self._mapping[n][0]
        k = displacement_field_index[field]
        i = self._mapping[n][1]
        j = self._mapping[n][2]

        for jj, derivatives_j in enumerate(((1, 0), (0, 1))):
            for kk, derivatives_k in enumerate(((1, 0), (0, 1))):
                derivatives = (derivatives_j[0] + derivatives_k[0], derivatives_j[1] + derivatives_k[1])
                ddu[k, jj, kk] = self._equations(xi1, xi2, field, i, j, *derivatives)

        if self.number_of_fields() == 6:
            ddv = ddu[3:6]
            ddu = ddu[0:3]
            return ddu, ddv
        else:
            return ddu

    def __call__(self, *args, **kwargs):
        u = args[0]
        xi1 = args[1]
        xi2 = args[2]

        result = np.zeros((self._number_of_fields,) + np.shape(xi1))
        for i in range(self.number_of_degrees_of_freedom()):
            result = result + self.shape_function(i, xi1, xi2, 0, 0) * u[i]

        if self.number_of_fields() == 6:
            v = result[3:6]
            u = result[0:3]
            return u, v
        else:
            return result

    def number_of_fields(self):
        return self._number_of_fields

