"""
Kernel Mutual Subspace Method
"""

# Authors: Junki Ishikawa

from sklearn.preprocessing import normalize as _normalize, LabelEncoder
import numpy as np
from scipy.linalg import block_diag

from .base_class import KernelSMBase, MSMInterface
from cvt.utils import rbf_kernel, dual_vectors, mean_square_singular_values


class KernelMSM(MSMInterface, KernelSMBase):
    """
    Kernel Mutual Subspace Method
    """

    def _get_gramians(self, X):
        """
        Parameters
        ----------
        X: array, (n_dims, n_samples)

        Returns
        -------
        G: array, (n_class, n_subdims, n_subdims)
            gramian matricies of references of each class
        """

        # _X, (n_dims, n_samples)
        # K, (n_samples, n_samples)
        K = rbf_kernel(X, X, self.sigma)
        # in_coeff, (n_samles, test_n_subdims)
        in_coeff, _ = dual_vectors(K, self.test_n_subdims)
        
        gramians = []
        for i in range(self.n_data):
            # ref_X, (n_dims, n_samples_ref_X)
            # ref_coeff, (n_samples_ref_X, n_subdims)
            ref_X, ref_coeff = self.dic[i]

            # _K, (n_samples_ref_X, n_samples)
            _K = rbf_kernel(ref_X, X, self.sigma)
            # S, (n_subdims, test_n_subdims)
            S = ref_coeff.T.dot(_K.dot(in_coeff))
            gramians.append(S)
        return np.array(gramians)


    def fast_predict_proba(self, X):
        """
        Predict class probabilities

        Parameters:
        -----------
        X: list of 2d-arrays, (n_vector_sets, n_samples, n_dims)
            List of input vector sets.

        Returns:
        --------
        pred: array, (n_vector_sets)
            Prediction array

        """

        n_input = len(X)
        n_ref =  len(self.dic)
        
        # preprocessing data matricies
        X = self._prepare_X(X)
        
        # manage reference informations
        ref_Xs, ref_coeffs = [], []
        for ref_X, ref_coeff in self.dic:
            ref_Xs.append(ref_X)
            ref_coeffs.append(ref_coeff)
    
        ref_mappings = np.array([i for i in range(len(ref_Xs)) for _ in range(ref_coeffs[i].shape[1])])
        ref_Xs = np.hstack(ref_Xs)
        ref_coeffs = block_diag(*ref_coeffs)
        
        in_coeffs = []
        for _X in X:
            K = rbf_kernel(_X, _X, self.sigma)
            in_coeff, _ = dual_vectors(K, self.test_n_subdims)
            in_coeffs.append(in_coeff)
        in_mappings = np.array([i for i in range(n_input) for _ in range(in_coeffs[i].shape[1])])
        in_Xs = np.hstack(X)
        in_coeffs = block_diag(*in_coeffs)
        
        K = rbf_kernel(in_Xs, ref_Xs, self.sigma)
        del ref_Xs, in_Xs
        
        S = in_coeffs.T.dot(K).dot(ref_coeffs)
        del in_coeffs, ref_coeffs, K
        
        # Split matrix into (n_input x n_ref) blocks
        in_split = np.where(np.diff(np.pad(in_mappings, (1, 0), 'constant')))[0]
        ref_split = np.where(np.diff(np.pad(ref_mappings, (1, 0), 'constant')))[0]
        S = [np.hsplit(_S, ref_split) for _S in np.vsplit(S, in_split)]
        
        vmssv = np.vectorize(lambda i, j: mean_square_singular_values(S[i][j]))
        pred = vmssv(*np.meshgrid(np.arange(n_input), np.arange(n_ref))).T
        
        del S, X
        return np.array(pred)
   
    