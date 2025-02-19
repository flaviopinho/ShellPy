import numpy as np
import pandas as pd
import sparse
from numbers import Integral, Real


class MultiIndex:
    """
    Defines a polynomial function (or vector, matrix, tensor).
    Example:
    U = 10 x1^2*x2*x3 - 5 x2 + 7*x1*x3^3
    x = [ -5, 3, 2 ]
    U(x) = 1205
    """
    def __init__(self, variable_dimension: int, result_shape=()):
        """
        Creates a MultiIndex entity.
        :param variable_dimension: Positive integer representing the number of polynomial variables.
        :param result_shape: Tuple defining the shape of the output value.
                             Examples:
                             - For a scalar function, use an empty tuple ().
                             - For a vector, use (n), where n is the vector's dimension.
                             - For a matrix, use (n, m), where n and m are the number of rows and columns.
        """

        # Verify if variable_dimension is positive integer
        if variable_dimension < 1:
            raise ValueError("variable_dimension must be a positive integer.")

        # Check if result_shape is a tuple
        if not isinstance(result_shape, tuple):
            raise ValueError("result_shape must be a tuple of integers.")

        # Verify that all elements in result_shape are positive integers
        for dim in result_shape:
            if not isinstance(dim, int) and dim < 1:
                raise ValueError("Each element in result_shape must be a positive integer.")

        # Initialization of class fields
        self.variable_dimension = variable_dimension
        self.result_shape = result_shape
        self.result_rank = len(result_shape)
        self._number_of_monomials = 0
        self._alpha = []
        self._coefficients = {}
        self._tensor = None

    def add_monomial(self, monomial_exponents, coefficient, coordinates=None) -> None:
        """
        Adds a monomial to the MultiIndex.
        :param monomial_exponents: List or tuple of size variable_dimension storing the exponents of the monomial.
        :param coefficient: Float representing the coefficient of the added monomial.
        :param coordinates: Position of the monomial in the result (optional for functional).
        """

        if len(monomial_exponents) != self.variable_dimension:
            raise ValueError("monomial_exponents dimension must be equal to variable_dimension.")

        for exponent in monomial_exponents:
            if not isinstance(exponent, Integral):
                raise ValueError("Each element in monomial_exponents must be a integer.")

        if coordinates is None:
            coordinates = []

        coordinates = list(coordinates)

        if len(coordinates) != self.result_rank:
            raise ValueError("Coordinate rank must be the same as result rank.")

        for k, coordinate in enumerate(coordinates):
            if coordinate >= self.result_shape[k] or coordinate < 0:
                raise ValueError("Coordinates dont mach the result_shape.")

        monomial_exponents = tuple(monomial_exponents)  # Convert to tuple for hashing
        j = self._alpha.index(monomial_exponents) if monomial_exponents in self._alpha else None

        if j is None:
            j = self._number_of_monomials
            self._alpha.append(monomial_exponents)
            self._number_of_monomials += 1

        coordinates.append(j)
        coordinate_tuple = tuple(coordinates)

        self._coefficients[coordinate_tuple] = self._coefficients.get(coordinate_tuple, np.float64(0)) + coefficient
        self._tensor = None

    def jacobian(self):
        """
        Return a MultiIndex instance that is the Jacobian of the current instance
        """
        jacob_shape = self.result_shape + (self.variable_dimension,)
        jacob_tensor = MultiIndex(self.variable_dimension, jacob_shape)

        for coordinate, value in self._coefficients.items():
            monomial_idx = coordinate[-1]
            monomial_exponents = self._alpha[monomial_idx]
            for idx_var, exponent in enumerate(monomial_exponents):
                if exponent == 0:
                    continue
                drop = exponent
                monomial_jacob = list(monomial_exponents)  # Avoid unnecessary copying
                monomial_jacob[idx_var] = drop - 1
                jacob_coefficient = value * drop
                jacob_coordinate = list(coordinate)  # Avoid unnecessary copying
                jacob_coordinate[-1] = idx_var
                jacob_tensor.add_monomial(monomial_jacob, jacob_coefficient, jacob_coordinate)

        return jacob_tensor

    @staticmethod
    def tensor_to_functional_multi_index(tensor):
        """
        Convert a tensor into a MultiIndex instance.

        :param tensor: A NumPy array or sparse tensor containing coefficients.
        :return: A MultiIndex instance.
        """
        if not isinstance(tensor, np.ndarray):
            raise ValueError("tensor must be a numpy array.")
        if len(set(tensor.shape)) != 1:
            raise ValueError("tensor must have all equal dimensions.")

        variable_dimension = tensor.shape[0]
        order = tensor.ndim

        indices = np.array(np.indices(tensor.shape).reshape(order, -1).T)
        coeffs = tensor.ravel()

        alpha = np.zeros((indices.shape[0], variable_dimension), dtype=int)
        for i in range(order):
            np.add.at(alpha, (np.arange(len(indices)), indices[:, i]), 1)

        # Usando Pandas para agrupar rapidamente os multi-índices e somar os coeficientes
        df = pd.DataFrame(alpha)
        df['coeff'] = coeffs
        grouped = df.groupby(df.columns[:-1].tolist(), sort=False, as_index=False).sum()

        unique_alpha = [tuple(row) for row in grouped.iloc[:, :-1].to_numpy()]
        summed_coeff = grouped['coeff'].to_numpy()

        multi_index = MultiIndex(variable_dimension)
        multi_index._number_of_monomials = len(unique_alpha)
        multi_index._alpha = unique_alpha
        multi_index._coefficients = {(i,): summed_coeff[i] for i in range(len(summed_coeff))}

        return multi_index

    @staticmethod
    def old_tensor_to_functional_multi_index(tensor):
        """
        Convert a tensor into a MultiIndex instance.

        :param tensor: A NumPy array or sparse tensor containing coefficients.
        :return: A MultiIndex instance.
        """
        if not isinstance(tensor, np.ndarray):
            raise ValueError("tensor must be a numpy array.")

        if len(set(tensor.shape)) != 1:
            raise ValueError("tensor must have all equal dimensions.")

        order = tensor.ndim
        variable_dimension = np.shape(tensor)[0]

        coeffs = tensor.ravel()
        repeated_exponents = list(np.ndindex(tensor.shape))
        alpha = [tuple(np.bincount(exponent, minlength=variable_dimension)) for exponent in repeated_exponents]

        # Identificar tuplas únicas em alpha
        unique_alpha, indices = np.unique(alpha, axis=0, return_inverse=True)
        unique_alpha = list(map(tuple, unique_alpha))

        # Somar os coefficients para tuplas repetidas
        summed_coeff = np.array([coeffs[indices == i].sum() for i in range(len(unique_alpha))])

        multi_index = MultiIndex(variable_dimension)
        multi_index._number_of_monomials = len(unique_alpha)
        multi_index._alpha = unique_alpha
        multi_index._coefficients = {(i,): summed_coeff[i] for i in range(len(summed_coeff))}

        return multi_index

    def __call__(self, vector_state):
        """
        Return the result of the MultiIndex for a given vector state
        Example:
        U = 10 x1^2*x2*x3 - 5 x2 + 7*x1*x3^3
        x = [ -5, 3, 2 ]
        U(x) = 1205
        :param vector_state: numpy array that contains the state of the variables
        :return: Result of the MultiIndex for the vector_state
        """
        if isinstance(vector_state, np.ndarray) and len(vector_state) != self.variable_dimension:
            raise ValueError("vector_state must be a np.array with dimension equal to variable_dimension.")

        alpha = np.array(self._alpha)
        U_alpha = np.prod(vector_state.T ** alpha, axis=1)
        if self._tensor is None:
            tensor_shape = self.result_shape + (self._number_of_monomials,)
            self._tensor = sparse.COO(self._coefficients, shape=tensor_shape)
        return sparse.tensordot(self._tensor, U_alpha, axes=1)



