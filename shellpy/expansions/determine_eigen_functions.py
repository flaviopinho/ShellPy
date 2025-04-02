import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
import scipy
from scipy.linalg import eigh
import mpmath as mp

high_precision_eigen_function = False

# Function to calculate eigenfunction expansion for given mid_surface_domain conditions
# Parameters:
# - boundary_conditions (tuple of str): A tuple containing two mid_surface_domain conditions for the two edges.
#   It can contain the values 'F' (free), 'S' (simply supported), or 'C' (clamped).
# - maximum_derivative (int, optional): The maximum number of derivatives to calculate for the eigenfunctions. Default is 3.
# - maximum_mode (int, optional): The maximum number of modes (eigenvalues) to calculate. Default is 20.
#
# Returns:
# - eigen_functions (dict): A dictionary where the keys are tuples of the form (mode, derivative),
#   and the values are lambda functions representing the corresponding eigenfunction and its derivative.
#   The eigenfunctions are calculated for the specified mid_surface_domain conditions and derivatives.

def determine_eigenfunctions(boundary_conditions, maximum_derivative=3, maximum_mode=20):
    # Handle mid_surface_domain condition for "free-free" case
    if boundary_conditions == ("F", "F"):
        return free_free(maximum_derivative, maximum_mode)

    # Handle mid_surface_domain condition for "simply supported" case
    if boundary_conditions == ("S", "S"):
        return ss_ss(maximum_derivative, maximum_mode)

    # Set the precision for mpmath calculations
    mp.mp.dps = 100

    # Define symbolic variables
    x_, C1_, C2_, C3_, C4_, k_ = sym.symbols('x_ C1_ C2_ C3_ C4_ k_')

    # General form for the eigenfunction expansion
    w = C1_ * sym.sin(k_ * x_) + C2_ * sym.cos(k_ * x_) + C3_ * sym.sinh(
        k_ * x_) + C4_ * sym.cosh(k_ * x_)

    # Initialize list to store mid_surface_domain condition equations
    BC_equations = []

    # Loop over both edges (0 and 1)
    for edge in (0, 1):
        # Check mid_surface_domain conditions and append appropriate equations
        if boundary_conditions[edge] == 'F':  # Free mid_surface_domain
            BC_equations.append(sym.diff(w, x_, 2).subs(x_, mp.mpf(edge)))  # 2nd derivative at edge
            BC_equations.append(sym.diff(w, x_, 3).subs(x_, mp.mpf(edge)))  # 3rd derivative at edge
        elif boundary_conditions[edge] == 'S':  # Simply supported mid_surface_domain
            BC_equations.append(w.subs(x_, mp.mpf(edge)))  # Function value at edge
            BC_equations.append(sym.diff(w, x_, 2).subs(x_, mp.mpf(edge)))  # 2nd derivative at edge
        elif boundary_conditions[edge] == 'C':  # Clamped mid_surface_domain
            BC_equations.append(w.subs(x_, mp.mpf(edge)))  # Function value at edge
            BC_equations.append(sym.diff(w, x_, 1).subs(x_, mp.mpf(edge)))  # 1st derivative at edge
        elif boundary_conditions[edge] == 'FC':  # Clamped mid_surface_domain
            BC_equations.append(sym.diff(w, x_, 3).subs(x_, mp.mpf(edge)))  # 3rd derivative at edge
            BC_equations.append(sym.diff(w, x_, 1).subs(x_, mp.mpf(edge)))  # 1st derivative at edge
        else:
            # Raise error if mid_surface_domain condition is invalid
            raise ValueError(f"Invalid boundary condition type: {boundary_conditions[edge]}")

    # Initialize matrix for mid_surface_domain condition equations
    BC_matrix = sym.zeros(4, 4)

    # Loop to fill the mid_surface_domain condition matrix
    for i in range(4):
        for j, var in enumerate((C1_, C2_, C3_, C4_)):
            BC_matrix[i, j] = sym.diff(BC_equations[i], var, 1)  # Differentiate with respect to coefficients

    # Calculate the determinant of the BC matrix
    det = sym.simplify(sym.det(BC_matrix))

    # Convert the determinant to a function for numerical evaluation
    lambda_det = sym.lambdify(k_, det, 'mpmath')
    lambda_det_prime = sym.lambdify(k_, det.diff(k_, 1), 'mpmath')

    # Initialize list for storing solution intervals
    solution_interval = []
    point1 = mp.mpf(0.01)  # Initial point for bisection
    value1 = lambda_det(point1)  # Calculate function value at point1

    dist = mp.mpf(1)  # Initial distance between intervals

    # Find intervals where the determinant changes sign (indicating eigenvalues)
    while len(solution_interval) <= maximum_mode:
        point2 = point1 + dist / 2  # New midpoint
        value2 = lambda_det(point2)  # Calculate function value at point2
        if value1 * value2 < 0:  # If the function changes sign, an eigenvalue exists between point1 and point2
            solution_interval.append((mp.mpf(point1), mp.mpf(point2)))

        # Adjust the distance between intervals for further calculations
        if len(solution_interval) > 1:
            dist = solution_interval[-1][0] - solution_interval[-2][1]

        # Move to the next point and calculate its function value
        point1 = point2
        value1 = value2

    # Solve for eigenvalues using Newton-Raphson method
    sol = []
    for interval in solution_interval:
        # sol.append(solver_bisection(lambda_det, *interval))  # Uncomment for bisection method
        sol.append(_solver_newton_raphson(lambda_det, lambda_det_prime, interval[0], interval[1]))

    # Solve for eigenvalues using Newton-Raphson method
    sol = []
    for interval in solution_interval:
        # sol.append(solver_bisection(lambda_det, *interval))  # Uncomment for bisection method
        sol.append(_solver_newton_raphson(lambda_det, lambda_det_prime, interval[0], interval[1]))

    eigen_functions = {}

    cont = 0
    if boundary_conditions == ("FC", "F") or boundary_conditions == ("F", "FC") or boundary_conditions == ("FC", "FC"):
        cont = 1
        for derivative in range(maximum_derivative):
            if derivative == 0:
                eigen_functions[(0, derivative)] = lambda x: np.ones(np.shape(x))
            else:
                eigen_functions[(0, derivative)] = lambda x: np.zeros(np.shape(x))

    if boundary_conditions == ("S", "F"):
        cont = 1
        for derivative in range(maximum_derivative):
            if derivative == 0:
                eigen_functions[(0, derivative)] = lambda x: x
            elif derivative == 1:
                eigen_functions[(0, derivative)] = lambda x: np.ones(np.shape(x))
            else:
                eigen_functions[(0, derivative)] = lambda x: np.zeros(np.shape(x))

    if boundary_conditions == ("F", "S"):
        cont = 1
        for derivative in range(maximum_derivative):
            if derivative == 0:
                eigen_functions[(0, derivative)] = lambda x: x-1
            elif derivative == 1:
                eigen_functions[(0, derivative)] = lambda x: np.ones(np.shape(x))
            else:
                eigen_functions[(0, derivative)] = lambda x: np.zeros(np.shape(x))


    # Calculate eigenfunctions for each eigenvalue and its derivatives
    for modo in range(maximum_mode):
        k = sol[modo]  # Eigenvalue

        # Substitute eigenvalue into the mid_surface_domain condition matrix
        M = mp.matrix(BC_matrix.subs(k_, k))

        # Solve eigenvalue problem for the matrix M
        eigvals, eigvectors = mp.eig(M)

        # Find the eigenvector corresponding to the smallest eigenvalue (near zero)
        idx = _find_nearest(eigvals, 0)
        Coeff = eigvectors[:, idx]  # Eigenvector coefficients
        Coeff = Coeff.apply(lambda x: x.real)  # Use real part of the coefficients
        Coeff = Coeff.apply(lambda x: mp.mpf(0) if mp.fabs(x) < mp.mpf(1e-30) else x)  # Threshold small values to zero

        # Normalize the eigenvector (L2 normalization)
        norm = mp.sqrt(sum(Coeff[i] ** 2 for i in range(len(Coeff))))  # Calculate the L2 norm
        Coeff = Coeff.apply(lambda x: x / norm)  # Normalize the coefficients

        # Simplify the expression for the eigenfunction
        ws = w.subs(((C1_, Coeff[0]), (C2_, Coeff[1]), (C3_, Coeff[2]), (C4_, Coeff[3]), (k_, k)))

        # Store the eigenfunction and its derivatives up to the maximum derivative order
        for derivative in range(maximum_derivative):
            if high_precision_eigen_function:
                eigen_functions[(modo+cont, derivative)] = sym.lambdify(x_, ws.diff(x_, derivative), 'mpmath')
            else:
                eigen_functions[(modo + cont, derivative)] = sym.lambdify(x_, ws.diff(x_, derivative), 'numpy')


    if high_precision_eigen_function:
        for key in eigen_functions:
            # Função lambdify original
            func = eigen_functions[key]

            # Criação de uma cópia da função lambdified e definição do comportamento
            def create_func_with_float_conversion(func):
                def func_with_float_conversion(*args):
                    # Caso args seja um array ou lista, iteramos sobre cada elemento
                    if isinstance(args[0], np.ndarray):  # Verificando se args[0] é um array
                        # Inicializa a estrutura de resultado com a mesma forma de args
                        result_mpf = np.empty_like(args[0], dtype=np.float64)

                        # Itera sobre cada elemento do ndarray mantendo a estrutura original
                        it = np.nditer(args[0], flags=['multi_index'])
                        for idx in it:
                            # Converte cada elemento de args para mpmath.mpf e aplica a função
                            result_mpf[it.multi_index] = float(func(mp.mpf(args[0][it.multi_index])))

                        return result_mpf
                    else:
                        # Se a entrada for um float ou int, tratamos normalmente
                        args_mpf = mp.mpf(args[0])  # Converter para mpmath.mpf

                        # Calcular o resultado com precisão mpmath e garantir que a saída seja do tipo float ou np.float64
                        result_mpf = float(func(args_mpf))  # Retorna o valor como float

                        return result_mpf

                return func_with_float_conversion

            # Atribuir a cópia da função modificada para a chave correspondente
            eigen_functions[key] = create_func_with_float_conversion(func)

    # Return the dictionary of eigenfunctions and their derivatives
    return eigen_functions


def free_free(maximum_derivative, maximum_mode):
    eigen_functions = {}

    def func(i, j, n):
        return lambda x: np.cos(i * np.pi * x - 1 / 2 * j * np.pi + 1 / 2 * np.pi * n) * (i * np.pi) ** n

    for derivative in range(maximum_derivative):
        if derivative == 0:
            eigen_functions[(0, derivative)] = lambda x: np.ones(np.shape(x))
        else:
            eigen_functions[(0, derivative)] = lambda x: np.zeros(np.shape(x))

    for derivative in range(maximum_derivative):
        if derivative == 0:
            eigen_functions[(1, derivative)] = lambda x: x
        elif derivative == 1:
            eigen_functions[(1, derivative)] = lambda x: np.ones(np.shape(x))
        else:
            eigen_functions[(1, derivative)] = lambda x: np.zeros(np.shape(x))

    for modo in range(maximum_mode):
        i = np.floor((modo + 2) / 2)
        j = (1 + (-1) ** (modo)) / 2
        for derivative in range(maximum_derivative):
            n = derivative
            eigen_functions[(modo+2, derivative)] = func(i, j, n)

    return eigen_functions


def ss_ss(maximum_derivative, maximum_mode):
    eigen_functions = {}

    def func(i, n):
        return lambda x: np.sin(i * np.pi * x + 1 / 2 * np.pi * n) * (i * np.pi) ** n

    for modo in range(maximum_mode):
        for derivative in range(maximum_derivative):
            n = derivative
            eigen_functions[(modo, derivative)] = func(modo + 1, derivative)

    return eigen_functions


def _solver_newton_raphson(f, f_prime, x0, x1, tol=mp.mpf(1e-30), max_iter=1000):
    a, b = x0, x1  # Definição do intervalo
    x = (a + b) / 2  # Começa no meio do intervalo

    for _ in range(max_iter):
        f_val = f(x)
        f_prime_val = f_prime(x)

        if abs(f_prime_val) < tol:  # Evita divisão por valores pequenos
            ValueError("Newton-Raphson error. Step too small.")
            return None

        x_next = x - f_val / f_prime_val  # Passo do Newton-Raphson

        # Se x_next estiver fora do intervalo, faz bisseção
        if x_next < a or x_next > b:
            x_next = (a + b) / 2  # Estratégia de fallback

        f_next = f(x_next)

        # Atualiza o intervalo preservando a raiz
        if f(a) * f_next < 0:
            b = x_next  # A raiz está entre a e x_next
        else:
            a = x_next  # A raiz está entre x_next e b

        # Critério de convergência
        if abs(b - a) < tol or abs(f_next) < tol:
            return x_next

        x = x_next  # Atualiza x

    print(f"Warning: Newton-Raphson did not converge within {max_iter} iterations.")
    return x  # Se não convergir


def _solver_bisection(f, a, b, tol=mp.mpf(1e-15), max_iter=100000):
    for _ in range(max_iter):
        c = (a + b) / 2
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        if abs(a - b) < tol:
            return c
    return c


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(np.real(array) - value)).argmin()
    return idx


def _remove_small_and_imaginary(matrix, threshold=mp.mpf(1e-10)):
    filtered = matrix.apply(lambda x: x if abs(x.imag) <= threshold else x.real)
    filtered = matrix.apply(lambda x: x if abs(x) > threshold else 0)
    return filtered


def _poly_eigen(*A):
    """
    Solve the polynomial eigenvalue problem:
        (A0 + e A1 +...+  e**p Ap)x = 0

    Return the eigenvectors [x_i] and eigenvalues [e_i] that are solutions.

    Usage:
        X, e = polyeig(A0, A1, ..., Ap)

    Most common usage, to solve a second-order system: (K + C e + M e**2) x = 0
        X, e = polyeig(K, C, M)
    """
    if len(A) <= 0:
        raise ValueError('Provide at least one matrix')
    for Ai in A:
        if Ai.shape[0] != Ai.shape[1]:
            raise ValueError('Matrices must be square')
        if Ai.shape != A[0].shape:
            raise ValueError('All matrices must have the same shape')

    n = A[0].shape[0]
    l = len(A) - 1
    # Assemble matrices for the generalized problem
    C = np.block([
        [np.zeros((n * (l - 1), n)), np.eye(n * (l - 1))],
        [-np.column_stack(A[0:-1])]
    ])
    D = np.block([
        [np.eye(n * (l - 1)), np.zeros((n * (l - 1), n))],
        [np.zeros((n, n * (l - 1))), A[-1]]
    ])
    # Solve the generalized eigenvalue problem
    e, X = scipy.linalg.eig(C, D)
    if np.all(np.isreal(e)):
        e = np.real(e)
    X = X[:n, :]

    # Sort eigenvalues/vectors
    # I = np.argsort(e)
    # X = X[:, I]
    # e = e[I]

    # Scale each mode by max
    X /= np.tile(np.max(np.abs(X), axis=0), (n, 1))

    return X, e