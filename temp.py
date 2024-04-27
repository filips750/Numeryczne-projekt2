from helpers import generate_random_vector, generate_random_matrix, dot_product, create_index_matrix, matrix_times_vector, add_two_matrixes


def get_diagonal(matrix):
    return [matrix[i][i] for i in range(len(matrix))]


def extract_lower(matrix):
    n = len(matrix)
    lower = [[0 if j >= i else matrix[i][j] for j in range(n)] for i in range(n)]
    return lower


def extract_upper(matrix):
    n = len(matrix)
    upper = [[0 if j <= i else matrix[i][j] for j in range(n)] for i in range(n)]
    return upper


def matrix_vector_multiplication(matrix, vector):
    rows = len(matrix)
    cols = len(matrix[0])
    result = [0] * rows
    for i in range(rows):
        result[i] = sum(matrix[i][j] * vector[j] for j in range(cols))
    return result


def subtract_vectors(vec1, vec2):
    return [vec1[i] - vec2[i] for i in range(len(vec1))]


def add_vectors(vec1, vec2):
    return [vec1[i] + vec2[i] for i in range(len(vec1))]


def divide_by_diagonal(vector, diag):
    return [vector[i] / diag[i] for i in range(len(vector))]


def divide_matrix_by_diagonal(matrix, diag):
    return [[matrix[i][j]/diag[i] for i in range(len(matrix))] for j in range(len(matrix))]


def vector_norm(vector):
    return (sum(x ** 2 for x in vector)) ** 0.5


def negate_vector(vector):
    return [-vector[i] for i in range(len(vector))]


def jacobi(A, b, tolerance=1e-12, max_iterations=1000):
    n = len(A)
    D = get_diagonal(A)
    L = extract_lower(A)
    U = extract_upper(A)
    x = [1 for _ in range(n)]
    M = divide_matrix_by_diagonal(add_two_matrixes(L, U), negate_vector(D))
    bm = divide_by_diagonal(b, D)

    for i in range(max_iterations):
        # Multiply L with x
        # L_x = matrix_vector_multiplication(L, x)
        # # Multiply U with x
        # U_x = matrix_vector_multiplication(U, x)
        # # Compute b - L*x - U*x
        # b_new = subtract_vectors(subtract_vectors(b, L_x), U_x)  # b - L*x - U*x
        # # Calculate the next iteration step
        # x_next = divide_by_diagonal(b_new, D)  # D^-1 * (b - (L + U) * x)
        # # Compute the residual vector

        x_old = x
        x = add_vectors(matrix_vector_multiplication(M, x_old), bm)
        residual = subtract_vectors(matrix_vector_multiplication(A, x), b)
        # Check convergence
        # print(vector_norm(residual))
        if vector_norm(residual) < tolerance:
            print(i)
            return x  # Converged solution
    return x


a, b = create_index_matrix(5)

print(b)

x = jacobi(a, b)
print(matrix_times_vector(a, x))
