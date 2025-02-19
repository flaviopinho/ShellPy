import time
import numpy as np

from multiindex import MultiIndex
from tensor_derivatives import tensor_derivative

if __name__ == "__main__":
    rank = 4
    dimension = 5
    shape = (dimension,) * rank

    u = np.random.rand(dimension)
    tensor = np.random.rand(*shape)

    start = time.time()
    multiindex1 = MultiIndex.tensor_to_functional_multi_index(tensor)
    stop = time.time()
    print('tempo1 = ', stop-start)

    print('functional')

    v1 = multiindex1(u)
    v = np.einsum('ijkl, i, j, k, l->', tensor, u, u, u, u)

    print(v, v1)

    print('force')

    F1 = multiindex1.jacobian()
    v1 = F1(u)

    d_tensor = tensor_derivative(tensor, 0)
    v = np.einsum('ijkl, j, k, l->i', d_tensor, u, u, u)

    print(v)
    print(v1)

    print('jacobian')

    J1 = F1.jacobian()
    v1 = J1(u)
    print(v1)

    dd_tensor = tensor_derivative(d_tensor, 1)
    v = np.einsum('ijkl, k, l->ij', dd_tensor, u, u)
    print(v)