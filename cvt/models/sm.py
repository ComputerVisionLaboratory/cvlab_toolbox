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

    def _predict(self, X):
        """
        Parameters
        ----------
        X: arrays, (n_dims, n_samples)
        """
        # self.dic shapes (n_classes, n_dims, n_subdims)
        # this line is projection to reference subspaces
        # proj, (n_classes, n_subdims, n_samples)
        proj = np.dot(self.dic.transpose(0, 2, 1), X)

        # length, (n_classes, n_samples)
        square_length = (proj**2).sum(axis=1)

        pred = np.argmax(square_length, axis=0)
        print(self.labels)
        pred = self.labels[pred]
        return pred
