"""
Kernel Mutual Subspace Method
"""

# Authors: Junki Ishikawa

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize as _normalize, LabelEncoder
import numpy as np

from cvt.utils import rbf_kernel, dual_vectors, mean_square_singular_values


class KernelMSM(BaseEstimator, ClassifierMixin):
    """
    Kernel Mutual Subspace Method

    Parameters
    ----------
    n_subdims : int, optional (default=3)
        A dimension of subspace. it must be smaller than the dimension of original space.

    normalize : boolean, optional (default=True)
        If this is True, all vectors are normalized as |v| = 1

    """

    def __init__(self, n_subdims=3, normalize=True, sigma=None, kernel_func=rbf_kernel):
        self.n_subdims = n_subdims
        self.normalize = normalize
        self.sigma = sigma
        self.kernel_func = kernel_func
        self.le = LabelEncoder()
        self.train_X = None
        self.labels = None
        self.A = None

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
        X: array-like, shape = [n_samples, n_vectors, n_features]
            Training vectors, where n_samples is number of samples, n_vectors is number of vectors on each samples
            and n_features is number of features. Since n_vectors may be variable on each samples, X can be lists
            containing n_sample matrices: [array(n_vectors{1}, n_features),..., array(n_vectors{n_samples}, n_features)]

        y: integer array, shape = [n_samples]
            Target values
        """

        # converted labels
        y = self.le.fit_transform(y)
        # number of classes
        n_classes = self.le.classes_.size
        # number of subspace dimension
        n_subdims = self.n_subdims

        if self.normalize:
            X = [_normalize(_X) for _X in X]

        self.train_X = X
        self.labels = y

        A = []
        for i in range(n_classes):
            # K is a Grammian matrix of all vectors, shape: (sum of n_vectors, sum of n_vectors)
            K = self.kernel_func(X[i], X[i], self.sigma)
            _A, _ = dual_vectors(K, n_subdims)
            A.append(_A)
        self.A = A

    def predict(self, X):
        """
        Predict each classes

        Parameters:
        -----------
        X: array-like, shape: (n_samples, n_vectors, n_features)
            Data Matrix

        Returns:
        --------
        pred: array-like, shape: (n_sample)
            Predictions

        """

        # number of classes
        n_classes = self.le.classes_.size
        n_subdims = self.n_subdims

        pred = []
        for _X in X:
            if self.normalize:
                _X = _normalize(_X)

            c = []
            for i in range(n_classes):
                K = self.kernel_func(_X, _X, self.sigma)
                A, _ = dual_vectors(K, n_subdims)
                train_X = self.train_X[i]
                _K = self.kernel_func(train_X, _X)
                S = self.A[i] @ _K @ A.T
                _c = mean_square_singular_values(S)
                c.append(_c)
            pred.append(self.labels[np.argmax(c)])

        return self.le.inverse_transform(pred)
