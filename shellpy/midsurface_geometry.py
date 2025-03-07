import numpy as np
import sympy as sym

from .cache_decorator import cache_method

# Define symbolic curvilinear coordinates
xi1_, xi2_ = sym.symbols('xi1 xi2')
curvilinear_coordinates_ = (xi1_, xi2_)


class MidSurfaceGeometry:
    """
    Defines the mid-surface of a thin shell and computes its differential geometry properties.
    Provides methods to compute fundamental geometric quantities, including:
        - position_vector
        - sqrtG (surface area element)
        - natural_base (covariant basis vectors)
        - reciprocal_base (contravariant basis vectors)
        - metric_tensor_covariant_components
        - metric_tensor_contravariant_components
        - curvature_tensor_covariant_components
        - curvature_tensor_mixed_components
        - christoffel_symbols
        - christoffel_symbols_first_derivative
    """

    def __init__(self, symbolic_position_vector):
        """
        Initializes the mid-surface geometry object.
        :param symbolic_position_vector: A sympy 3x1 matrix defining the parametric surface, dependent on xi1_ and xi2_.
        Example:
        R = [xi1; xi2; xi1**2 + xi2**2]
        symbolic_position_vector = sym.Matrix([xi1_, xi2_, xi1_ ** 2 + xi2_ ** 2])
        """

        with sym.evaluate(False):
            zero = xi1_ * 0 + xi2_ * 0  # Ensure symbolic zero retains dependencies

        # Define position vector
        self.R_ = symbolic_position_vector + sym.Matrix([zero, zero, zero])
        for i in range(3):
            self.R_[i, 0] = zero + self.R_[i, 0]
        self._position_vector = sym.lambdify((xi1_, xi2_), self.R_, 'numpy')

        # Compute covariant basis vectors
        M1 = sym.diff(self.R_, xi1_, 1)  # Partial derivative w.r.t xi1
        M2 = sym.diff(self.R_, xi2_, 1)  # Partial derivative w.r.t xi2

        # Compute normal vector and surface area element
        Maux = M1.cross(M2)
        sqrtG = sym.simplify(sym.sqrt(Maux.dot(Maux))) + zero  # Determinant of metric tensor
        M3 = Maux / sqrtG  # Normalized normal vector

        # Compute contravariant basis vectors
        MR1 = sym.Matrix(sym.simplify(M2.cross(M3) / sqrtG))
        MR2 = sym.Matrix(sym.simplify(-M1.cross(M3) / sqrtG))
        MR3 = sym.Matrix(M3)

        # Ensure symbolic consistency
        for i in range(3):
            M1[i, 0] = zero + M1[i, 0]
            M2[i, 0] = zero + M2[i, 0]
            M3[i, 0] = zero + M3[i, 0]
            MR1[i, 0] = zero + MR1[i, 0]
            MR2[i, 0] = zero + MR2[i, 0]
            MR3[i, 0] = zero + MR3[i, 0]

        # Store numerical functions for fast evaluation
        self._sqrtG = sym.lambdify((xi1_, xi2_), sqrtG, 'numpy')
        self._M1 = sym.lambdify((xi1_, xi2_), M1, 'numpy')
        self._M2 = sym.lambdify((xi1_, xi2_), M2, 'numpy')
        self._M3 = sym.lambdify((xi1_, xi2_), M3, 'numpy')
        self._MR1 = sym.lambdify((xi1_, xi2_), MR1, 'numpy')
        self._MR2 = sym.lambdify((xi1_, xi2_), MR2, 'numpy')
        self._MR3 = sym.lambdify((xi1_, xi2_), MR3, 'numpy')

        # Define base vectors
        natural_base_ = (M1, M2, M3)
        reciprocal_base_ = (MR1, MR2, MR3)

        # Compute first derivatives of the basis vectors
        natural_base_first_derivative_ = {}
        reciprocal_base_first_derivative_ = {}
        for i in range(3):
            for alpha in range(2):
                natural_base_first_derivative_[(i, alpha)] = sym.diff(natural_base_[i], curvilinear_coordinates_[alpha])
                reciprocal_base_first_derivative_[(i, alpha)] = sym.diff(reciprocal_base_[i],
                                                                         curvilinear_coordinates_[alpha])

        # Compute second derivatives of the basis vectors
        natural_base_second_derivative_ = {}
        reciprocal_base_second_derivative_ = {}
        for i in range(3):
            for beta in range(2):
                for alpha in range(2):
                    natural_base_second_derivative_[(i, beta, alpha)] = sym.diff(
                        sym.diff(natural_base_[i], curvilinear_coordinates_[beta]), curvilinear_coordinates_[alpha])
                    reciprocal_base_second_derivative_[(i, beta, alpha)] = sym.diff(
                        sym.diff(reciprocal_base_[i], curvilinear_coordinates_[beta]), curvilinear_coordinates_[alpha])

        # Compute metric tensor components
        metric_tensor_covariant_components_ = sym.matrices.zeros(2, 2)
        metric_tensor_contravariant_components_ = sym.matrices.zeros(2, 2)

        metric_tensor_covariant_components_extended_ = sym.matrices.zeros(3, 3)
        metric_tensor_contravariant_components_extended_ = sym.matrices.zeros(3, 3)

        curvature_tensor_covariant_components_ = sym.matrices.zeros(2, 2)
        curvature_tensor_mixed_components_ = sym.matrices.zeros(2, 2)

        for alpha in range(2):
            for beta in range(2):
                metric_tensor_covariant_components_[alpha, beta] = natural_base_[alpha].dot(natural_base_[beta])
                metric_tensor_contravariant_components_[alpha, beta] = reciprocal_base_[alpha].dot(
                    reciprocal_base_[beta])
                curvature_tensor_covariant_components_[alpha, beta] = zero + natural_base_[alpha].dot(
                    natural_base_first_derivative_[(2, beta)])
                curvature_tensor_mixed_components_[alpha, beta] = zero + reciprocal_base_[alpha].dot(
                    natural_base_first_derivative_[(2, beta)])


        # Store numerical functions for fast evaluation
        self._metric_tensor_covariant_components = sym.lambdify((xi1_, xi2_),
                                                                metric_tensor_covariant_components_,
                                                                'numpy')
        self._metric_tensor_contravariant_components = sym.lambdify((xi1_, xi2_),
                                                                    metric_tensor_contravariant_components_,
                                                                    'numpy')
        self._curvature_tensor_covariant_components = sym.lambdify((xi1_, xi2_),
                                                                   curvature_tensor_covariant_components_,
                                                                   'numpy')
        self._curvature_tensor_mixed_components = sym.lambdify((xi1_, xi2_),
                                                               curvature_tensor_mixed_components_,
                                                               'numpy')

        # Compute Christoffel symbols (first kind)
        christoffel_symbols_ = sym.matrices.zeros(3 * 3 * 2, 1)
        cont = 0
        for i in range(3):
            for j in range(3):
                for alpha in range(2):
                    if i == 2 and j == 2:
                        christoffel_symbols_[cont, 0] = zero
                    else:
                        christoffel_symbols_[cont, 0] = zero + reciprocal_base_[i].dot(
                            natural_base_first_derivative_[(j, alpha)])
                    cont += 1

        self._chris1 = sym.lambdify((xi1_, xi2_), christoffel_symbols_, 'numpy')

        # Compute Christoffel symbols first derivatives
        christoffel_symbols_first_derivative_ = sym.matrices.zeros(3 * 3 * 2 * 2, 1)
        cont = 0
        for i in range(3):
            for j in range(3):
                for alpha in range(2):
                    for beta in range(2):
                        # (C^{i}_{j alpha})_{,beta}
                        if i == 2 and j == 2:
                            christoffel_symbols_first_derivative_[cont, 0] = zero
                            cont += 1
                        else:
                            christoffel_symbols_first_derivative_[cont, 0] = zero + (
                                    reciprocal_base_first_derivative_[(i, beta)].dot(
                                        natural_base_first_derivative_[(j, alpha)]) +
                                    reciprocal_base_[i].dot(
                                        natural_base_second_derivative_[(j, alpha, beta)]))
                            cont += 1

        self._chris2 = sym.lambdify((xi1_, xi2_), christoffel_symbols_first_derivative_, 'numpy')

        self.cache = {}  # Cache for optimization

    @cache_method
    def sqrtG(self, xi1, xi2):
        """
        Computes the square root of the determinant of the metric tensor (Jacobian determinant).
        Used to determine the differential element of the mid-surface.
        """
        return self._sqrtG(xi1, xi2)

    @cache_method
    def natural_base(self, xi1, xi2):
        """
        Returns the natural (covariant) basis vectors of the mid-surface at a given point.
        These vectors are tangent to the surface.
        """
        return self._natural_base(xi1, xi2)

    @cache_method
    def reciprocal_base(self, xi1, xi2):
        """
        Returns the reciprocal (contravariant) basis vectors of the mid-surface at a given point.
        These vectors satisfy the reciprocal relation with the natural basis.
        """
        return self._reciprocal_base(xi1, xi2)

    @cache_method
    def metric_tensor_covariant_components(self, xi1, xi2):
        """
        Computes the covariant components of the metric tensor.
        These components describe the inner product of the natural basis vectors.
        """
        return self._metric_tensor_covariant_components(xi1, xi2)

    @cache_method
    def metric_tensor_contravariant_components(self, xi1, xi2):
        """
        Computes the contravariant components of the metric tensor.
        These components are the inverse of the covariant metric tensor.
        """
        return self._metric_tensor_contravariant_components(xi1, xi2)

    @cache_method
    def metric_tensor_covariant_components_extended(self, xi1, xi2):
        """
        Computes the covariant components of the metric tensor.
        These components describe the inner product of the natural basis vectors.
        """
        G = np.zeros((3, 3) + np.shape(xi1))
        G[0:2,0:2] = self._metric_tensor_covariant_components(xi1, xi2)
        G[2, 2] = 1
        return G

    @cache_method
    def metric_tensor_contravariant_components_extended(self, xi1, xi2):
        """
        Computes the contravariant components of the metric tensor.
        These components are the inverse of the covariant metric tensor.
        """
        G = np.zeros((3, 3) + np.shape(xi1))
        G[0:2,0:2] = self._metric_tensor_contravariant_components(xi1, xi2)
        G[2, 2] = 1
        return G

    @cache_method
    def curvature_tensor_covariant_components(self, xi1, xi2):
        """
        Computes the covariant components of the curvature tensor.
        These describe how the surface bends in space.
        """
        return self._curvature_tensor_covariant_components(xi1, xi2)

    @cache_method
    def curvature_tensor_mixed_components(self, xi1, xi2):
        """
        Computes the mixed components of the curvature tensor.
        These are obtained by contracting the curvature tensor with the contravariant metric tensor.
        """
        return self._curvature_tensor_mixed_components(xi1, xi2)

    @cache_method
    def christoffel_symbols(self, xi1, xi2):
        """
        Computes the Christoffel symbols of the second kind.
        These symbols represent the connection coefficients used in the covariant derivative.
        """
        return self._christoffel_symbols(xi1, xi2)

    @cache_method
    def christoffel_symbols_first_derivative(self, xi1, xi2):
        """
        Computes the first derivative of the Christoffel symbols.
        These derivatives appear in the expression of the Riemann curvature tensor.
        """
        return self._christoffel_symbols_first_derivative(xi1, xi2)

    @cache_method
    def gaussian_curvature(self, xi1, xi2):
        curvature = self.curvature_tensor_mixed_components(xi1, xi2)
        shape = list(range(curvature.ndim))

        _gaussian_curvature = np.linalg.det(curvature.transpose(shape[2:] + shape[0:2]))

        return _gaussian_curvature

    def mean_curvature(self, xi1, xi2):
        curvature = self.curvature_tensor_mixed_components(xi1, xi2)

        _mean_curvature = 0.5 * np.einsum('aa...->...', curvature)

        return _mean_curvature

    @cache_method
    def shifter_tensor(self, xi1, xi2, xi3):
        delta = np.eye(2, 2)
        curvature = self.curvature_tensor_mixed_components(xi1, xi2)
        result = np.einsum('abxy,z->abxyz', curvature, xi3)

        delta_expanded = np.einsum('ab, ...->ab...', delta, np.ones(np.shape(xi1)+np.shape(xi3)))

        return result + delta_expanded

    @cache_method
    def shifter_tensor_inverse(self, xi1, xi2, xi3):
        shifter = self.shifter_tensor(xi1, xi2, xi3)  # Obtém o tensor shifter
        shape = list(range(shifter.ndim))

        a = shape[0]
        b = shape[1]

        shape[0] = shape[-2]
        shape[1] = shape[-1]

        shape[-2] = a
        shape[-1] = b

        shifter_inv = np.linalg.inv(shifter.transpose(shape)).transpose(shape)

        return shifter_inv

    @cache_method
    def shifter_tensor_inverse_approximation(self, xi1, xi2, xi3):
        delta = np.eye(2, 2)
        delta_expanded = np.einsum('ab,...->ab...', delta, np.ones(np.shape(xi1)+np.shape(xi3)))

        curvature = self.curvature_tensor_mixed_components(xi1, xi2)
        aux1 = np.einsum('...,z->...z', curvature, xi3)
        aux2 = np.einsum('ag...z, gb..., z->ab...z', aux1, curvature, xi3)
        aux3 = np.einsum('ao...z, ob..., z->ab...z', aux2, curvature, xi3)
        aux4 = np.einsum('ao...z, ob..., z->ab...z', aux3, curvature, xi3)

        return delta_expanded - aux1 + aux2 - aux3 + aux4

    @cache_method
    def determinant_shifter_tensor(self, xi1, xi2, xi3):
        trace_K = 2 * self.mean_curvature(xi1, xi2)
        gaussian_curvature = self.gaussian_curvature(xi1, xi2)
        shifter_det = (np.ones(np.shape(xi1) + np.shape(xi3)) +
                       np.einsum('xy, z->xyz', trace_K, xi3) +
                       np.einsum('xy, z->xyz', gaussian_curvature, xi3 ** 2))

        return shifter_det


    def __call__(self, xi1, xi2):
        """
        Allows the instance of MidSurfaceGeometry to be called as a function.
        This method returns the position vector of the mid-surface at specific values
        of the curvilinear coordinates xi1 and xi2.

        :param xi1: The first curvilinear coordinate (e.g., xi1 in the parametric surface).
        :param xi2: The second curvilinear coordinate (e.g., xi2 in the parametric surface).

        :return: The position vector R at the specified values of xi1 and xi2.
        """

        # Call the precomputed lambdified function for position vector with xi1 and xi2
        return self._position_vector(xi1, xi2)

    def _natural_base(self, xi1, xi2):
        return self._M1(xi1, xi2).squeeze(), self._M2(xi1, xi2).squeeze(), self._M3(xi1, xi2).squeeze()

    def _reciprocal_base(self, xi1, xi2):
        return self._MR1(xi1, xi2).squeeze(), self._MR2(xi1, xi2).squeeze(), self._MR3(xi1, xi2).squeeze()

    def _christoffel_symbols(self, xi1, xi2):
        return np.array(self._chris1(xi1, xi2)).reshape((3, 3, 2) + np.shape(xi1))

    def _christoffel_symbols_first_derivative(self, xi1, xi2):
        return np.array(self._chris2(xi1, xi2)).reshape((3, 3, 2, 2) + np.shape(xi1))
