import sympy as sym
import numpy as np

from displacement_expansion import DisplacementExpansion, displacement_field_index
from mid_surface_domain import RectangularMidSurfaceDomain


class EnrichedCosineExpansion(DisplacementExpansion):
    """
    Define a displacement expansion using enriched cosine functions
    """

    def __init__(self, expansion_size, rectangular_domain: RectangularMidSurfaceDomain, boundary_conditions, mapping=None):
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

        u = np.zeros((3,) + np.shape(xi1), dtype=np.float64)
        vec = self._mapping[n]
        field = self._mapping[n][0]
        k = displacement_field_index[field]
        i = vec[1]
        j = vec[2]
        c = self._equations(xi1, xi2, field, i, j, derivative1, derivative2)
        u[k] = c
        return u

    def shape_function_first_derivatives(self, n, xi1, xi2):
        """
        Determine the first derivative of the shape function with respect to xi1 and xi2
        :param n: index of the n-th shape function
        :param xi1: curvilinear coordinate 1
        :param xi2: curvilinear coordinate 2
        :return: Returns a matrix (numpy ndarray) that contains the derivative of the shape_functions
        """
        du = np.zeros((3, 2) + np.shape(xi1), dtype=np.float64)
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
        :param n: index of the n-th shape function
        :param xi1: curvilinear coordinate 1
        :param xi2: curvilinear coordinate 2
        :return: Returns a tensor (numpy ndarray) that contains the derivative of the shape_functions
        """
        ddu = np.zeros((3, 2, 2) + np.shape(xi1), dtype=np.float64)
        field = self._mapping[n][0]
        k = displacement_field_index[field]
        i = self._mapping[n][1]
        j = self._mapping[n][2]

        for jj, derivatives_j in enumerate(((1, 0), (0, 1))):
            for kk, derivatives_k in enumerate(((1, 0), (0, 1))):
                derivatives = (derivatives_j[0] + derivatives_k[0], derivatives_j[1] + derivatives_k[1])
                ddu[k, jj, kk] = self._equations(xi1, xi2, field, i, j, *derivatives)

        return ddu

    def _determine_equations(self):

        cosine_set_functions = {}
        for field_, bc in self._boundary_conditions.items():
            for direction, BC in bc.items():
                if field_ == 'u3':
                    cosine_set_functions[(field_, direction)] = EnrichedCosineExpansion._generate_set_C1(BC, self._edges[direction], 3, 30)
                else:
                    cosine_set_functions[(field_, direction)] = EnrichedCosineExpansion._generate_set_C0(BC, self._edges[direction], 3, 30)

        equations = lambda xi1, xi2, field, i, j, derivative1, derivative2: (
                cosine_set_functions[field, "xi1"][(i - 1, derivative1)](xi1)
                * cosine_set_functions[field, "xi2"][(j - 1, derivative2)](xi2)
        )
        return equations

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

    def __call__(self, *args, **kwargs):
        u = args[0]
        xi1 = args[1]
        xi2 = args[2]

        result = np.zeros((3,) + np.shape(xi1))
        for i in range(self.number_of_degrees_of_freedom()):
            result = result + self.shape_function(i, xi1, xi2, 0, 0) * u[i]

        return result

    @staticmethod
    def _generate_set_C0(boundary_conditions, boundary, maximum_derivative=3, maximum_mode = 10):
        xi_, j_, x_= sym.symbols('xi_ j_ x_')

        a = boundary[0]
        b = boundary[1]
        L = b - a

        xi_ = (2 * x_ - (a+b))/L

        Fc = []
        Fc.append(sym.simplify(1/4*(3*xi_+1)*(xi_-1)))
        Fc.append(sym.simplify(1/4*(3*xi_-1)*(xi_+1)))
        Fcj = sym.simplify(sym.cos(sym.pi/2*(j_-3)*(xi_+1))-Fc[0]+((-1) ** j_)*Fc[1])

        functions_set = {}
        for derivative in range(maximum_derivative):
            count = 0
            for point in range(2):
                if boundary_conditions[point] == "F":
                    functions_set[(count, derivative)] = sym.lambdify(x_, Fc[point].diff(x_, derivative))
                    count += 1
                elif boundary_conditions[point] != "S":
                    print('erro')

            for point in range(3, maximum_mode):
                functions_set[(count, derivative)] = sym.lambdify(x_, sym.simplify(Fcj.subs(j_, point).diff(x_, derivative)))
                count += 1

        return functions_set

    @staticmethod
    def _generate_set_C1(boundary_conditions, boundary, maximum_derivative = 3, maximum_mode = 10):
        xi_, j_, x_= sym.symbols('xi_ j_ x_')

        a = boundary[0]
        b = boundary[1]
        L = b - a

        xi_ = (2 * x_ - (a+b))/L

        Fc = []
        Fc.append(sym.simplify(-1 / 16 * (3 * xi_ + 1) * (5 * xi_ + 7) * (xi_ - 1) ** 2))
        Fc.append(sym.simplify(L / 32 * (xi_ + 1) * (5 * xi_ + 1) * (xi_ - 1) ** 2))
        Fc.append(sym.simplify(-1 / 16 * (3 * xi_ - 1) * (5 * xi_ - 7) * (xi_ + 1) ** 2))
        Fc.append(sym.simplify(L / 32 * (xi_ - 1) * (5 * xi_ - 1) * (xi_ + 1) ** 2))
        Fcj = sym.simplify(sym.cos(sym.pi/2*(j_-5)*(xi_+1))-Fc[0]+((-1) ** j_)*Fc[2])

        functions_set = {}
        for derivative in range(maximum_derivative):
            count = 0
            for point in range(2):
                if boundary_conditions[point] == "F":
                    functions_set[(count, derivative)] = sym.lambdify(x_, Fc[2*point].diff(x_, derivative))
                    count += 1
                    functions_set[(count, derivative)] = sym.lambdify(x_, Fc[2*point+1].diff(x_, derivative))
                    count += 1

                elif boundary_conditions[point] == "S":
                    functions_set[(count, derivative)] = sym.lambdify(x_, Fc[2*point+1].diff(x_, derivative))
                    count += 1
                elif boundary_conditions[point] != "C":
                    print('erro')

            for point in range(5, maximum_mode):
                functions_set[(count, derivative)] = sym.lambdify(x_, sym.simplify(Fcj.subs(j_, point).diff(x_, derivative)))
                count += 1

        return functions_set






