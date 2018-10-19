"""
Mutual Subspace Method
"""

# Authors: Junki Ishikawa

import numpy as np

from .base_class import SMBase, MSMInterface
from cvt.utils import subspace_bases, mean_square_singular_values


class MutualSubspaceMethod(MSMInterface, SMBase):
    """
    Mutual Subspace Method
    """

    def _predict(self, X):
        """
        Parameters
        ----------
        X: list of 2d-arrays, (n_vector_sets, n_dims, n_samples)
        """

        pred = []
        for _X in X:
            # bases, (n_dims, n_subdims)
            bases = subspace_bases(_X, self.n_subdims)

            # grammians, (n_classes, n_subdims, n_subdims)
            grammians = np.dot(self.dic.transpose(0, 2, 1), bases)

            # i_th singular value of grammian of subspace bases is
            # square root of cosine of i_th cannonical angles
            # average of square of them is caonnonical angle between subspaces
            c = [mean_square_singular_values(g) for g in grammians]
            pred.append(self.labels[np.argmax(c)])
        pred = np.array(pred)
        return pred
