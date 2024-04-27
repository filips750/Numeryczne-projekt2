import copy
import time
from helpers import matrix_times_vector, dot_product, forward_substitution, backward_substitution, vec_minus_vec, float_times_vector, create_index_matrix, generate_random_matrix, generate_random_vector


def LU(A, b):
    L, U = LU_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x


def LU_decomposition(A):
    m = len(A)
    L = [[1 if i == j else 0 for i in range(m)] for j in range(m)]
    U = copy.deepcopy(A)
    for i in range(m):
        for j in range(i + 1, m):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            U[j] = vec_minus_vec(U[j], float_times_vector(factor, U[i]))
    return L, U


create_index_matrix(7)
A = [
    [2, 1, 1, 0],
    [4, 3, 3, 1],
    [8, 7, 9, 5],
    [6, 7, 9, 8]
]

b = [1, 2, 3, 4]

L, U = LU_decomposition(A)

x = LU(A, b)
print(x)
print("B: ", matrix_times_vector(A, x))

# testA = generate_random_matrix(500)
# testb = generate_random_vector(500)
# a = time.time()
# x = LU(testA, testb)
# b = time.time() - a
# print(b)
