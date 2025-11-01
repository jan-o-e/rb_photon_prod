import numpy as np
from qutip import Qobj, tensor, destroy, qeye


def create_custom_tensor(
    dims, non_zero_indices_values, N, destroy_1=False, destroy_2=False
):
    """
    Create a custom tensor Qobj with specified non-zero elements and optional destruction operators.

    Parameters:
    dims (tuple): Dimensions of the matrix, e.g., (6, 6).
    non_zero_indices_values (list of tuples): List of tuples where each tuple contains
                                              the indices of the non-zero element and its value,
                                              e.g., [((2, 2), 1), ((3, 3), 1)].
    N (int): Dimension for qeye and destroy operators.
    destroy_1 (bool): Whether the first operator should be a destroy operator.
    destroy_2 (bool): Whether the second operator should be a destroy operator.

    Returns:
    Qobj: The constructed quantum object tensor.
    """
    # Initialize a zero matrix of given dimensions
    matrix = np.zeros(dims)

    # Set the specified non-zero elements
    for (i, j), value in non_zero_indices_values:
        matrix[i, j] = value

    # Convert the matrix to Qobj
    qobj_matrix = Qobj(matrix)

    # Determine the operators based on the boolean flags
    operator_1 = destroy(N) if destroy_1 else qeye(N)
    operator_2 = destroy(N) if destroy_2 else qeye(N)

    # Construct the first part of the tensor product
    a_return = tensor(qobj_matrix, operator_1, operator_2)

    # Define and add the second part of the tensor product
    id_matrix = np.identity(dims[0])
    # Set the specified non-zero elements
    for (i, j), value in non_zero_indices_values:
        id_matrix[i, j] = 0

    qobj_additional_matrix = Qobj(id_matrix)
    a_return += tensor(qobj_additional_matrix, qeye(N), qeye(N))

    return a_return
