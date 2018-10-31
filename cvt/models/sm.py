"""
Subspace Method
"""

# Authors: Junki Ishikawa

import numpy as np

from .base_class import SMBase
from cvt.utils import subspace_bases, mean_square_singular_values


class SubspaceMethod(SMBase):
    """
    Mutual Subspace Method
    """

    def _predict_proba(self, X):
        """
        Parameters
        ----------
        X: arrays, (n_dims, n_samples)
        """
        # self.dic shapes (n_classes, n_dims, n_subdims or greater)
        # this line is projection to reference subspaces
        # proj, (n_classes, n_subdims, n_samples)
        dic = self.dic[:, :, :self.n_subdims]
        proj = np.dot(dic.transpose(0, 2, 1), X)

        # length, (n_classes, n_samples)
        length = np.sqrt((proj**2).sum(axis=1))
        # proba, (n_samples, n_classes)
        proba = (length / np.linalg.norm(X, axis=0)).T
        return proba
