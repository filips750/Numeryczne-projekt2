import copy
import random
import time
from helpers import generate_random_vector, generate_random_matrix, diag, matrix_times_vector, forward_substitution, backward_substitution, vec_plus_vec, vec_minus_vec, float_times_vector, create_index_matrix, tril, triu, get_diag, add_two_matrixes, negate, norm


def divide_by_diagonal(matrix, diag_elements):
    divided = [[matrix[i][j] / diag_elements[i] for j in range(len(matrix[0]))] for i in range(len(matrix))]
    return divided


def forward_substitution_matrix(L, B):
    n = len(B)
    m = len(B[0])
    Y = [[0] * m for _ in range(n)]
    for col in range(m):
        b = [B[i][col] for i in range(n)]
        for i in range(n):
            sum1 = sum(L[i][j] * Y[j][col] for j in range(i))
            Y[i][col] = (b[i] - sum1) / L[i][i]
    return Y


def calculate_M(L, U, D):
    diag_elements = diag(D)
    L_plus_U = add_two_matrixes(L, U)
    divided = divide_by_diagonal(L_plus_U, diag_elements)
    M = negate(divided)
    return M


def Jacobi(a, b, max_iterations=1000):
    u = triu(a)
    l = tril(a)
    d = get_diag(a)
    bm = forward_substitution(add_two_matrixes(d, l), b)
    M = calculate_M(l, u, d)
    x = [1 for _ in range(len(a))]
    err_norm = 1
    iterations = 0
    while (err_norm > 1e-6 and iterations < max_iterations):
        x_old = copy.deepcopy(x)
        x = vec_plus_vec(matrix_times_vector(M, x_old), bm)
        err_norm = norm(x)
        iterations += 1
    return x


# a, b = generate_random_matrix(10), generate_random_vector(10)
a, b = create_index_matrix(10)
# a = [
#     [2, 1, 1, 0],
#     [4, 3, 3, 1],
#     [8, 7, 9, 5],
#     [6, 7, 9, 8]
# ]

# b = [1, 2, 3, 4]
x = Jacobi(a, b)
print(b)
print(matrix_times_vector(a, b))


def GaussSeidel(a, b, max_iterations=1000):
    u = triu(a)
    l = tril(a)
    d = get_diag(a)
    bm = forward_substitution(add_two_matrixes(d, l), b)
    M = negate(forward_substitution_matrix(add_two_matrixes(d, l), u))
    iterations = 0
    err_norm = 1
    x = [1 for _ in range(len(a))]
    while (err_norm > 1e-6 and iterations < max_iterations):
        x_old = copy.deepcopy(x)
        x = vec_plus_vec(matrix_times_vector(M, x_old), bm)
        err_norm = norm(x)
        iterations += 1
    return x


x = GaussSeidel(a, b)
print(b)
print(matrix_times_vector(a, b))
