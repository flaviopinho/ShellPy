import numpy as np
from numpy.polynomial.legendre import Legendre

from ..expansions.simple_expansions import fourier_expansion_for_periodic_solutions


# ==================================================
# Legendre 1D expansion on arbitrary interval
# ==================================================

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

            def f(x, P=Pk, d=derivative):
                return (2 / L) ** d * P(map_to_reference(x))

            functions[(k, derivative)] = np.vectorize(f)

    return functions


# ==================================================
# EAS Expansion
# ==================================================

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
        """
        Tensor product basis:
        phi_ij(xi1, xi2) = f_i(xi1) * f_j(xi2)
        """
        field, i, j = self._mapping[n]

        f1 = self._basis[(field, "xi1")][(i, 0)]
        f2 = self._basis[(field, "xi2")][(j, 0)]

        return f1(xi1) * f2(xi2)

    # --------------------------------------------------
    # Internals
    # --------------------------------------------------

    def _set_mapping(self):
        """
        Mapping now INCLUDES mode 0 (constant mode)
        """
        mapping = []

        for field, (m, n) in self._expansion_size.items():
            for i in range(0, m):
                for j in range(0, n):
                    mapping.append((field, i, j))

        return mapping

    def _build_basis(self):
        """
        Build 1D bases per field and direction.
        Legendre for non-periodic BCs
        Fourier for ("R","R")
        """
        base = {}

        for field, bc in self._boundary_conditions.items():

            modes = [t for t in self._mapping if t[0] == field]

            if modes:
                max_i = max(t[1] for t in modes)
                max_j = max(t[2] for t in modes)
            else:
                max_i = 0
                max_j = 0

            for direction, BC in bc.items():

                if direction == "xi1":
                    maximum_mode = max_i
                else:
                    maximum_mode = max_j

                if BC == ("R", "R"):
                    base[(field, direction)] = fourier_expansion_for_periodic_solutions(
                        self._edges[direction],
                        maximum_derivative=1,
                        maximum_mode=maximum_mode + 2
                    )
                else:
                    base[(field, direction)] = legendre_expansion_on_interval(
                        self._edges[direction],
                        maximum_derivative=0,
                        maximum_mode=maximum_mode + 1
                    )

        return base
