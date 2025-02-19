import numpy as np


def tensor_derivative(tensor, n=0):
    """
    Calculates the derivative of the tensor T by summing its permutations based on the base index n.

    :param tensor: Input tensor (numpy ndarray).
    :param n: Base index for the permutation.
    :return: Derived tensor.
    """
    rank = tensor.ndim  # Number of dimensions of the tensor
    tensor_jacobian = np.copy(tensor)  # Copy the original tensor

    for i in range(n+1, rank):  # Iterate over the indices to permute
        p = list(range(rank))  # Create a list of indices [0, 1, ..., rank-1]
        p[i], p[n] = p[n], p[i]  # Swap the indices i and n
        tensor_jacobian += np.transpose(tensor, axes=p)  # Apply the permutation and add it to the tensor

    return tensor_jacobian
