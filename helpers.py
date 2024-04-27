import math
import random


def add_two_matrixes(a, b):
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("Number of columns in the matrix must match the number of elements in the vector.")
    return [
        [a[i][j] + b[i][j] for i in range(len(a))]
        for j in range(len(a))
    ]


def negate(x):
    return [
        [-x[i][j] for i in range(len(x))]
        for j in range(len(x))
        ]


def vec_minus_vec(vec1, vec2):
    return [a - b for a, b in zip(vec1, vec2)]


def vec_plus_vec(vec1, vec2):
    return [a + b for a, b in zip(vec1, vec2)]


def float_times_vector(scalar, vec):
    return [scalar * x for x in vec]


def matrix_times_vector(matrix, vector):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    if len(vector) != num_cols:
        raise ValueError("Number of columns in the matrix must match the number of elements in the vector.")
    result = [0] * num_rows
    for i in range(num_rows):
        result[i] = sum(matrix[i][j] * vector[j] for j in range(num_cols))
    return result


def dot_product(mat1, mat2):
    rows1 = len(mat1)
    cols1 = len(mat1[0])
    rows2 = len(mat2)
    cols2 = len(mat2[0])
    if cols1 != rows2:
        raise ValueError("Matrix multiplication not possible. Columns of first matrix must equal rows of second matrix.")
    result = [[0 for _ in range(cols2)] for _ in range(rows1)]
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result


def forward_substitution(L, b):
    n = len(b)
    y = [0] * n
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i][j] * y[j]
        y[i] = (b[i] - sum) / L[i][i]
    return y


def diag(matrix):
    return [matrix[i][i] for i in range(len(matrix))]


def divide_by_diagonal(matrix, diag_elements):
    divided = [[matrix[i][j] / diag_elements[i] for j in range(len(matrix[0]))] for i in range(len(matrix))]
    return divided


def backward_substitution(U, y):
    n = len(y)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        sum = 0
        for j in range(i + 1, n):
            sum += U[i][j] * x[j]
        x[i] = (y[i] - sum) / U[i][i]
    return x


def get_diag(A):
    return [
        [0 if i != j else A[i][j] for i in range(len(A))]
        for j in range(len(A))
        ]


def triu(A):
    return [
        [0 if i - 1 < j else A[i][j] for i in range(len(A))]
        for j in range(len(A))
    ]


def tril(A):
    return [
        [0 if i > j - 1 else A[i][j] for i in range(len(A))]
        for j in range(len(A))
    ]


def create_index_matrix(m):
    a1 = 5 + 5
    a2 = -1
    a3 = -1
    a = [[a1 if i == j else 0 for i in range(m)] for j in range(m)]
    for i in range(m-1):
        a[i+1][i] = a2
        a[i-1][i] = a2
    a[m-1][0] = 0
    a[m-2][m-1] = a2
    for i in range(2, m):
        a[i-2][i] = a3
        a[i][i-2] = a3
    f = 3
    b = [math.sin((i * f+1)) for i in range(m)]
    return a, b


def generate_random_matrix(n):
    return [[random.random() for _ in range(n)] for _ in range(n)]


def generate_random_vector(n):
    return [random.random() for _ in range(n)]


def norm(vector):
    return math.sqrt(sum(x ** 2 for x in vector))
