"""
Mathematical utilities
"""

# Authors: Junki Ishikawa
#          Takahiro Inagaki

import numpy as np
from scipy.linalg import eigh


def mean_square_singular_values(X):
    """
    calculate mean square of singular values of X

    Parameters:
    -----------
    X : array-like, shape: (n, m)

    Returns:
    --------
    c: mean square of singular values
    """

    # _, s, _ = np.linalg.svd(X)
    # mssv = (s ** 2).mean()

    # Frobenius norm means square root of
    # sum of square singular values
    mssv = (X * X).sum() / min(X.shape)
    return mssv


def max_square_singular_values(X):
    """
    calculate mean square of singular values of X

    Parameters:
    -----------
    X : array-like, shape: (n, m)

    Returns:
    --------
    c: mean square of singular values
    """

    _, s, _ = np.linalg.svd(X)
    mssv = (s ** 2).max()
    return mssv


def canonical_angle(X, Y):
    """
    Calculate cannonical angles beween subspaces

    Parameters
    ----------
    X: basis matrix, array-like, shape: (n_subdim_X, n_dim)
    Y: basis matrix, array-like, shape: (n_subdim_Y, n_dim)

    Returns
    -------
    c: float, similarity of X, Y

    """

    return mean_square_singular_values(Y @ X.T)


def canonical_angle_matrix(X, Y):
    """Calculate canonical angles between subspaces
    example     similarity = MathUtils.calc_basis_vector(X, Y)

    Parameters
    ----------
    X: set of basis matrix, array-like, shape: (n_set_X, n_subdim, n_dim)
        n_subdim can be variable on each subspaces
    Y: set of basis matrix, array-like, shape: (n_set_Y, n_subdim, n_dim)
        n_set can be variable from n_set of X
        n_subdim can be variable on each subspaces

    Returns:
        C: similarity matrix, array-like, shape: (n_set_X, n_set_Y)

    """

    n_set_X, n_set_Y = len(X), len(Y)
    C = np.zeros((n_set_X, n_set_Y))
    for x in range(n_set_X):
        for y in range(n_set_Y):
            C[x, y] = canonical_angle(X[x], Y[y])

    return C


# faster method
def canonical_angle_matrix_f(X, Y):
    """Calculate canonical angles between subspaces
    example     similarity = MathUtils.calc_basis_vector(X, Y)

    Parameters
    ----------
    X: set of basis matrix, array-like, shape: (n_set_X, n_subdim, n_dim)
        n_subdim can be variable on each subspaces
    Y: set of basis matrix, array-like, shape: (n_set_Y, n_subdim, n_dim)
        n_set can be variable from n_set of X
        n_subdim can be variable on each subspaces

    Returns:
        C: similarity matrix, array-like, shape: (n_set_X, n_set_Y)

    """
    X = np.transpose(X, (0, 2, 1))
    D = np.dot(Y, X)
    _D = np.transpose(D, (0, 2, 1, 3))
    _, C, _ = np.linalg.svd(_D)
    sim = C**2
    return sim.mean(2)


def _eigh(X, eigvals=None):
    """
    A wrapper function of numpy.linalg.eigh and scipy.linalg.eigh

    Parameters
    ----------
    X: array-like, shape (a, a)
        target symmetric matrix
    eigvals: tuple, (lo, hi)
        Indexes of the smallest and largest (in ascending order) eigenvalues and corresponding eigenvectors
        to be returned: 0 <= lo <= hi <= M-1. If omitted, all eigenvalues and eigenvectors are returned.

    Returns
    -------
    e: array-like, shape (a) or (n_dims)
        eigenvalues with descending order
    V: array-like, shape (a, a) or (a, n_dims)
        eigenvectors
    """

    if eigvals != None:
        e, V = eigh(X, eigvals=eigvals)
    else:
        # numpy's eigh is faster than scipy's when all calculating eigenvalues and eigenvectors
        e, V = np.linalg.eigh(X)

    e, V = e[::-1], V[:, ::-1]

    return e, V


def _eigen_basis(X, eigvals=None):
    """
    Return subspace basis using PCA

    Parameters
    ----------
    X: array-like, shape (a, a)
        target matrix
    n_dims: integer
        number of basis

    Returns
    -------
    e: array-like, shape (a) or (n_dims)
        eigenvalues with descending order
    V: array-like, shape (a, a) or (a, n_dims)
        eigenvectors
    """

    try:
        e, V = _eigh(X, eigvals=eigvals)
    except np.linalg.LinAlgError:
        # if it not converges, try with tiny salt
        salt = 1e-8 * np.eye(X.shape[0])
        e, V = eigh(X + salt, eigvals=eigvals)

    return e, V


def _get_eigvals(n, n_subdims, higher):
    """
    Culculate eigvals for eigh

    Parameters
    ----------
    n: int
    n_subdims: int, dimension of subspace
    higher: boolean, if True, use higher `n_subdim` basis

    Returns
    -------
    eigvals: tuple of 2 integers
    """

    if n_subdims is None:
        return None

    if higher:
        low = max(0, n - n_subdims)
        high = n - 1
    else:
        low = 0
        high = min(n - 1, n_subdims - 1)

    return low, high


def subspace_bases(X, n_subdims=None, higher=True, return_eigvals=False):
    """
    Return subspace basis using PCA

    Parameters
    ----------
    X: array-like, shape (n_dimensions, n_vectors)
        data matrix
    n_subdims: integer
        number of subspace dimension
    higher: bool
        if True, this function returns eigenvectors collesponding
        top-`n_subdims` eigenvalues. default is True.
    return_eigvals: bool
        if True, this function also returns eigenvalues.
    Returns
    -------
    V: array-like, shape (n_dimensions, n_subdims)
        bases matrix
    w: array-like shape (n_subdims)
        eigenvalues
    """

    if X.shape[0] <= X.shape[1]:
        eigvals = _get_eigvals(X.shape[0], n_subdims, higher)
        # get eigenvectors of autocorrelation matrix X @ X.T
        w, V = _eigen_basis(X @ X.T, eigvals=eigvals)
    else:
        # use linear kernel to get eigenvectors
        A, w = dual_vectors(X.T @ X, n_subdims=n_subdims, higher=higher)
        V = X @ A

    if return_eigvals:
        return V, w
    else:
        return V


def dual_vectors(K, n_subdims=None, higher=True, eps=1e-6):
    """
    Calc dual representation of vectors in kernel space

    Parameters:
    -----------
    K :  array-like, shape: (n_samples, n_samples)
        Grammian Matrix of X: K(X, X)
    n_subdims: int, default=None
        Number of vectors of dual vectors to return
    higher: boolean, default=None
        If True, this function returns eigenbasis corresponding to 
            higher `n_subdims` eigenvalues in descending order.
        If False, this function returns eigenbasis corresponding to 
            lower `n_subdims` eigenvalues in descending order.
    eps: float, default=1e-20
        lower limit of eigenvalues

    Returns:
    --------
    A : array-like, shape: (n_samples, n_samples)
        Dual replesentation vectors.
        it satisfies lambda[i] * A[i] @ A[i] == 1, where lambda[i] is i-th biggest eigenvalue
    e:  array-like, shape: (n_samples, )
        Eigen values descending sorted
    """

    eigvals = _get_eigvals(K.shape[0], n_subdims, higher)
    e, A = _eigen_basis(K, eigvals=eigvals)

    # replace if there are too small eigenvalues
    e[(e < eps)] = eps

    A = A / np.sqrt(e)

    return A, e


def cross_similarities(refs, inputs):
    """
    Calc similarities between each reference spaces and each input subspaces

    Parameters:
    -----------
    refs: list of array-like (n_dims, n_subdims_i)
    inputs: list of array-like (n_dims, n_subdims_j)

    Returns:
    --------
    similarities: array-like, shape (n_refs, n_inputs)
    """

    similarities = []
    for _input in inputs:
        sim = []
        for ref in refs:
            sim.append(mean_square_singular_values(ref.T @ _input))
        similarities.append(sim)

    similarities = np.array(similarities)

    return similarities
