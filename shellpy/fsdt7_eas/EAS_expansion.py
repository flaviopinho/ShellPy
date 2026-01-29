import numpy as np
from numpy.polynomial.legendre import Legendre
from shellpy.expansions.simple_expansions import fourier_expansion_for_periodic_solutions


class EasExpansion:

    def __init__(self, expansion_size, rectangular_domain, boundary_conditions):
        self._expansion_size = expansion_size
        self._edges = rectangular_domain.edges
        self._boundary_conditions = boundary_conditions

        self._mapping = self._set_mapping()
        self._number_of_fields = len(expansion_size)

        self._basis = self._build_basis()

    # --------------------------------------------------
    # API mínima
    # --------------------------------------------------

    def number_of_degrees_of_freedom(self):
        return len(self._mapping)

    def shape_function(self, n, xi1, xi2):
        field, i, j = self._mapping[n]

        phi = (
            self._basis[(field, "xi1")][(i - 1, 0)](xi1)
            * self._basis[(field, "xi2")][(j - 1, 0)](xi2)
        )

        return phi

    # --------------------------------------------------
    # Internals
    # --------------------------------------------------

    def _build_basis(self):

        basis = {}

        for field, bc in self._boundary_conditions.items():

            modes = [t for t in self._mapping if t[0] == field]
            max_i = max(t[1] for t in modes)
            max_j = max(t[2] for t in modes)

            for direction, BC in bc.items():

                max_mode = max_i if direction == "xi1" else max_j
                edge = self._edges[direction]

                if BC == ("R", "R"):
                    basis[(field, direction)] = fourier_expansion_for_periodic_solutions(
                        edge, 1, max_mode + 1)
                else:
                    basis[(field, direction)] = legendre_expansion_on_interval(
                        edge, maximum_derivative=0, maximum_mode=max_mode)

        return basis

    def _set_mapping(self):
        mapping = []
        for field, (m, n) in self._expansion_size.items():
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    mapping.append((field, i, j))
        return mapping


def legendre_expansion_on_interval(boundary, maximum_derivative, maximum_mode):
    a, b = boundary
    L = b - a

    def map_to_reference(x):
        return 2 * (x - a) / L - 1

    functions = {}

    for derivative in range(maximum_derivative + 1):
        for k in range(maximum_mode):

            Pk = Legendre.basis(k)

            if derivative > 0:
                Pk = Pk.deriv(derivative)

            def f(x, P=Pk):
                return (2 / L) ** derivative * P(map_to_reference(x))

            functions[(k, derivative)] = np.vectorize(f)

    return functions
