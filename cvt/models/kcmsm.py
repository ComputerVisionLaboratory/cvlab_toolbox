"""
Kernel Constrained Mutual Subspace Method
"""

# Authors: Junki Ishikawa

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize as _normalize, LabelEncoder
import numpy as np
from scipy.linalg import block_diag

from cvt.utils import rbf_kernel, dual_vectors, mean_square_singular_values, subspace_bases, cross_similarities


class KernelCMSM(BaseEstimator, ClassifierMixin):
    """
    Kernel Constrained Mutual Subspace Method

    Parameters
    ----------
    n_subdims : int
        A dimension of subspace. it must be smaller than the dimension of original space.

    kgds_rate : int
        A rate of dimension of Kernel GDS.
        This model uses int(len(X) * kgds_rate) dimensions KGDS

    normalize : boolean, optional (default=True)
        If this is True, all vectors are normalized as |v| = 1

    kernel_func : function, optional (default=rbf_kernel)
        A function which takes 2 matrix and returns gram matrix

    sigma : float, optional (default=None)
        A parameter of rbf_kernel
    """

    def __init__(self, n_subdims=5, kgds_rate=0.9, normalize=True, sigma=None, n_jobs=1, ignore_check_dimensions=False):
        self.n_subdims = n_subdims
        self.kgds_rate = kgds_rate
        self.normalize = normalize
        self.sigma = sigma
        self.kernel_func = rbf_kernel
        self.n_jobs = n_jobs
        self.ignore_check_dimensions = ignore_check_dimensions
        self.le = LabelEncoder()
        self.train_X = None
        self.labels = None
        self.mappings = None
        self.W = None
        self.refs = None

    def get_params(self, deep=True):
        return {'n_subdims': self.n_subdims, 'sigma': self.sigma, 'kernel_func': self.kernel_func}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """
        Fit the model according to the given traininig data and parameters

        Parameters
        ----------
        X: array-like, shape = (n_samples, n_vectors, n_features)
            Training vectors, where n_samples is number of samples, n_vectors is number of vectors on each samples
            and n_features is number of features. Since n_vectors may be variable on each samples, X can be lists
            containing n_sample matrices: [array(n_vectors{1}, n_features),..., array(n_vectors{n_samples}, n_features)]

        y: integer array, shape = (n_samples, )
            Target values
        """

        if not self.ignore_check_dimensions:
            self.__check_dimensions(X)

        # converted labels
        y = self.le.fit_transform(y)
        # number of classes
        n_classes = self.le.classes_.size
        # numbers of each subspace dimension
        n_subdims = [min(len(_X), self.n_subdims) for _X in X]
        # an array which maps a vector to its label
        mappings = np.array([y[i] for i, _X in enumerate(X) for _ in range(len(_X))])

        if self.normalize:
            X = [_normalize(_X) for _X in X]

        # Data matrix, shape: (sum of n_vectors, n_features)
        X = np.vstack(X)

        # (sum of n_vectors, n_features) -> (n_features, sum of n_vectors)
        X = X.T

        self.train_X = X
        self.labels = y
        self.mappings = mappings

        # K is a Grammian matrix of all vectors, shape: (sum of n_vectors, sum of n_vectors)
        K = self.kernel_func(X.T, X.T, self.sigma)
        K = (K + K.T) / 2   # make it strictly simentry

        # A has dual vectors of each class in its diagonal
        E = []
        for i in range(n_classes):
            p = (mappings == y[i])
            # clipping K(X[y==c], X[y==c])
            _K = K[p][:, p]
            _A, _ = dual_vectors(_K, n_subdims[i])
            E.append(_A)
        E = block_diag(*E)

        # D is a matrix, shape: (n_classes * n_subdims, n_classes * n_subdims)
        D = E.T @ K @ E

        # Dual vectors over all classes
        hi = int(D.shape[0] * self.kgds_rate)
        eigvals = (0, min(hi, D.shape[0]-1))
        B, _ = dual_vectors(D, eigvals=eigvals, truncate=1e-18)
        W = B.T @ E.T

        X_kgds = W @ K
        X_kgds = [X_kgds[:, mappings == y[i]] for i in range(n_classes)]

        # reference subspaces
        refs = self.__subspace_bases(X_kgds, n_subdims=n_subdims)

        self.W = W
        self.refs = refs

    def predict(self, X):
        """
        Predict each classes

        Parameters:
        -----------
        X: array-like, shape: (n_samples, n_vectors, n_features)
            Data Matrix

        Returns:
        --------
        pred: array-like, shape: (n_samples, )
            Predictions

        """

        n_data = len(X)
        n_subdims = [min(len(_X), self.n_subdims) for _X in X]

        mappings = np.array([i for i, _X in enumerate(X) for _ in range(len(_X))])

        if self.normalize:
            X = [_normalize(_X) for _X in X]

        X = np.vstack(X).T

        K = self.kernel_func(self.train_X.T, X.T)
        X_kgds = self.W @ K
        X_kgds = [X_kgds[:, mappings == i] for i in range(n_data)]

        # input subspaces
        inputs = self.__subspace_bases(X_kgds, n_subdims=n_subdims)

        # similarities per references
        similarities = cross_similarities(self.refs, inputs)

        pred = self.labels[np.argmax(similarities, axis=1)]

        return self.le.inverse_transform(pred)

    def __subspace_bases(self, X, n_subdims):
        """
        Fit the model according to the given traininig data and parameters

        Parameters
        ----------
        X: array-like, shape = (n_samples, n_vectors, n_features)
            Training vectors, where n_samples is number of samples, n_vectors is number of vectors on each samples
            and n_features is number of features. Since n_vectors may be variable on each samples, X can be lists
            containing n_sample matrices: [array(n_vectors{1}, n_features),..., array(n_vectors{n_samples}, n_features)]

        y: integer array, shape = (n_samples, )
            Target values
        """

        return [subspace_bases(_X, n_subdims[i]) for i, _X in enumerate(X)]

    def __check_dimensions(self, X):
        ids = np.where(list(map(lambda _X: len(_X) < self.n_subdims, X)))[0]

        raise Exception(f'''
            Some of data matrix has less dimensions than n_subdims={self.n_subdims}
            matrix index is {ids}
        ''')
