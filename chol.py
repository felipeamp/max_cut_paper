#!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np


_FINFO = np.finfo(float)
_EPS = _FINFO.eps


def is_invertible(A):
    return A.shape[0] == A.shape[1] and np.linalg.matrix_rank(A) == A.shape[0]


def chol_higham(A, eps=_EPS):
    """
    Pivoting Cholesky Decompostion
    using algorithm by Nicholas J. Higham, 2008

    Parameters
    ----------
    A : array_like
        Input matrix.
    eps : float
        Tollerence value for the eigen values. Values smaller than
        A.shape[0] * tol * np.diag(A).max() are considered to be zero.
    Return `(P, L, E)` such that `P.T * A * P = L * L.T - E`.

    Returns
    -------
    P : np.ndarray
        Permutation matrix
    R : np.ndarray
        Upper triangular decompostion
    E : np.ndarray
        Error matrix

    Examples
    --------
    >>> A = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
    >>> P, L, E = chol_higham(A)
    >>> P, L = np.matrix(P), np.matrix(L)
    >>> print np.diag(E)
    [ 0.00000000e+00   0.00000000e+00   0.00000000e+00]
    >>> print np.allclose(P.T*A*P, L*L.T-E)
    True
    """

    A = np.asfarray(A).copy()
    B = np.asfarray(A).copy()
    a0 = np.diag(A).max()
    assert len(A.shape) == 2
    n = A.shape[0]

    R = np.zeros((n, n))
    piv = np.arange(n)

    for k in range(n):
        q = np.argmax(np.diag(A[k:, k:])) + k
        if A[q, q] <= np.abs(n * a0 * eps): # stopping condition
            if k == 0:
                raise ValueError("Negative definite matrix")
            # The code below is only for positive-definite matrices. For singular ones, it should be
            # zero.
            # for j in range(k, n):
            #     R[j, j] = R[j-1, j-1]/float(j)
            break

        tmp = A[:, k].copy()
        A[:, k] = A[:, q]
        A[:, q] = tmp

        tmp = R[:, k].copy()
        R[:, k] = R[:, q]
        R[:, q] = tmp

        tmp = A[k, :].copy()
        A[k, :] = A[q, :]
        A[q, :] = tmp

        piv[k], piv[q] = piv[q], piv[k]

        R[k, k] = np.sqrt(A[k, k])
        r = A[k, k+1:]/R[k, k]
        R[k, k+1:] = r
        A[k+1:, k+1:] -= np.outer(r, r)

    P = np.zeros((n, n))
    for k in range(n):
        P[k, piv[k]] = 1.0

    E = np.dot(R.T, R) - np.dot(np.dot(P.T, B), P)

    return P, R.T, E


def test():
    orig = np.array([[1, 1, 0], [0, 0, 0.6], [0, 0, 0.8]], dtype=float)
    y = np.dot(orig.T, orig)

    print('-' * 80)
    print('TESTING chol_higham')
    P, L, E = chol_higham(y)
    print('orig:')
    print(orig)
    print('y')
    print(y)
    print('P')
    print(P)
    print('P.T')
    print(P.T)
    print('P.T*y*P')
    print(np.dot(np.dot(P.T, y), P))
    print('L*L.T')
    print(np.dot(L, L.T))
    print('E')
    print(E)
    print('np.diag(E)')
    print(np.diag(E))
    print('np.allclose')
    print(np.allclose(np.dot(np.dot(P.T, y), P), np.dot(L, L.T) - E))
    print('orig:')
    print(orig)
    print('found:')
    print(np.dot(L.T, P.T))

if __name__ == '__main__':
    test()
