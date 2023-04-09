import numpy as np

"""
Funções utilizadas:
np.dot: produto escalar.
np.zeros(n): cria um vetor de dimensão n, preenchido por zeros.
np.array: define um vetor.
o índice :k, num vetor ou lista, indica todos os valores de zero até k, excluindo k.
"""

# Fatoração LU
def gauss_lu_factorization(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(n):
        L[k, k] = 1
        for i in range(k, n):
            U[k, i] = A[k, i] - np.dot(L[k, :k], U[:k, i])
        for i in range(k+1, n):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

    return L, U

# Solução do sistema linear
def solve_linear_system(L, U, b):
    n = len(L)
    y = np.zeros(n)
    x = np.zeros(n)

    # Resolvendo L*y = b, obtendo y
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    # Resolvendo U*x = y, obtendo x
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

# Matrizes de exemplo
A = np.array([[2, -1, 0], [1, 3, 1], [-1, 2, 3]])
b = np.array([2, 5, 3])

L, U = gauss_lu_factorization(A)

x = solve_linear_system(L, U, b)

print("Solucao para Ax=b: ", x)