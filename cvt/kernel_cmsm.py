# -*- coding: utf-8 -*-
'''
Created on Thu Oct. 5 00:08 2017

@author: utscgonionys

All dimension of matrix is converted as follow
MATLAB: X is a cell(N, 1), where, N=\sum_{c=1}^{n_class}{N_c}., N_c is a number of c-class data. 
        X{i} = [n_dim, n_set]

Python: X is a list. len(X) == N. where, N=\sum_{c=1}^{n_class}{N_c}, N_c is a number of c-class data. 
        X[i] = [n_dim, n_set], np.ndarray.
        n_set is variable.
'''


import numpy as np

class KernelCMSM:
    '''
    @brief   Class for KCMSM
    
    -----functions:
    __init__: Initialize parameters
    fit(data_list, label): Training funciton
    vectors_gds_projection(data_matrix): KGDS projection function
    -----
    '''

    def __init__(self, n_subdim, n_sigma):
        '''
        Initialize parameters
        kcmsm = KernelCMSM(n_subdim, n_sigma)
        ---------
        n_subdim: Reference subspace dimension. Temporarily, This is a scalar greater than 1.
        n_sigma: Gausian kernel parameter. k(x, y) = exp(||x - y||^2 / (2 * n_sigma^2))
        ---------
        '''  

        self.n_subdim = n_subdim
        self.n_sigma = n_sigma

        
    def fit(self, data_list, label):
        '''
        Apply KPCA and Generate KGDS
        kcmsm.fit(data_list, label)
        ---------
        data_list: Including data matrix. data_list[k] is k-th data matrix. data_list[k] is (n_feature_dimension x n_set) matrix.
        label: This i-th element indicates to which class data_list[i] belongs. Type of label is np.ndarray. Range of label's value is [0, n_class - 1].
        ---------
        '''  

        self.n_dim = data_list[0].shape[0]
        self.n_class = len(np.unique(label))        
        self.X = np.transpose(np.hstack(data_list[:]))
       
        # calculate kernel matrix
        self.K = MathUtils.calc_kernelgram_matrix(self.X, self.X, sigma=self.n_sigma)
        
        # E is a matrix, which is used to construct the D. About D, please see Fukui-sensei's TPAMI paper.
        E = np.zeros((self.K.shape[0], self.n_subdim*self.n_class))

        # 
        i_subdim = np.arange(0, E.shape[1]+ self.n_subdim, self.n_subdim)
        i_data = np.hstack((0, np.cumsum([len(np.where(label==i)[0]) for i in range(self.n_class)])))
        
        sub_vec = []
        
        # KPCA + calculate E
        for c in range(0, self.n_class):
            ## KPCA phase

            ### Extract class kernel matrix from entire kernel matrix.
            Kc = self.K[i_data[c]:i_data[c+1], i_data[c]:i_data[c+1]]
            [eig_val, eig_vec] = np.linalg.eigh(Kc)

            ### Sort eigenvalues in descending order and extract eigenvector as many as the requested number(n_subdim).            
            eig_id = np.argsort(eig_val)[::-1]
            eig_id = eig_id[0:self.n_subdim]
            eig_val = eig_val[eig_id]
            eig_vec = eig_vec[:,eig_id]

            sub_vec.append(eig_vec / np.sqrt(eig_val))

            ## E is a block diagonal matrix
            E[i_data[c]:i_data[c+1], i_subdim[c]:i_subdim[c+1]] = sub_vec[c]

        # Generate KGDS 
        D = np.dot(np.transpose(E), self.K)
        D = np.dot(D, E)
        [eig_val, gds_basis] = np.linalg.eigh(D)
        
        ## Delete eigenvalues lower than 0.
        ind = np.squeeze(np.asarray(np.where(eig_val > 0)))
        eig_val = eig_val[ind];
        gds_basis = gds_basis[:, ind]

        ## Sort eigenvalues in ascending order
        eig_id = np.argsort(eig_val)
        eig_val = eig_val[eig_id]
        gds_basis = gds_basis[:, eig_id]

        self.gds_basis = np.transpose(gds_basis / np.sqrt(eig_val))
        self.eig_val = eig_val
        self.sub_basis = sub_vec
        self.E = np.transpose(E)

    def vectors_gds_projection(self, data_matrix):
        '''
        Apply KGDS projection
        projected_data = kcmsm.vectors_gds_projecton(data_matrix)
        -----------
        data_matrix: 2D-matrix (n_feature_dim x n_data)
        -----------
        '''

        K = MathUtils.calc_kernelgram_matrix(self.X, np.transpose(data_matrix), sigma=self.n_sigma)
        ret = np.dot(self.gds_basis, self.E)

        return np.dot(ret, K)



def get_clsdata(data_list, label, i):
    '''
    Get the i-th class data from list.
    i_th_class_data = get_clsdata(data_list, label, i)
    --------------
    data_list: Including data matrix. data_list[k] is k-th data matrix. data_list[k] is (n_feature_dimension x n_set) matrix.
    label: Type of label is np.ndarray. Range of label's value is [0, n_class - 1].
    i: object class number. Also, range of this value is [0, n_class - 1].
    --------------
    '''

    ind = list(np.where(label == i)[0])
    ret = np.hstack([data_list[x] for x in ind])
    return ret


"""
****************************************************************************
****************************************************************************
****************************************************************************
Following Class and functions are extracted from cvtpython tools box.
****************************************************************************
****************************************************************************
****************************************************************************

"""
def size(matrix, dim=0):
    """ Get a matrix size
    example   A = np.array([[[1, 2, 3, 3, 3], [4, 5, 6, 6, 6]], 
          [[10, 11, 12, 6, 6], [13, 14, 15, 7, 7]]])
            size(A, dim=1) # 2
            size(A) # (2, 2, 5)

    Args:
        matrix (ndarray): the matrix
        dim (int, optional): the dimension. if dim=0 return the shape of matrix

    Returns:
        size: The value of size or the shape of matrix
    """
    if dim != 0:
        mat_size = matrix.shape
        return mat_size[dim - 1]
    else:
        mat_size = matrix.shape
        if len(mat_size) == 1:
            return mat_size[0]
        return mat_size

class MathUtils:
    @staticmethod
    def calc_l2distance_matrix(X, Y):
        """Calculate L2-norm distance between 2 matrices 
        example     d = MathUtils.calc_l2distance_matrix(X, Y)

        Args:
            X (TYPE): 2-D matrix
            Y (TYPE): 2-D matrix

        Returns:
            distance_matrix: 

        """
        [n_sample1, n_dim1] = size(X)
        [n_sample2, n_dim2] = size(Y)
        A = np.sum(X * X, 1).reshape([1, n_sample1])
        B = np.sum(Y * Y, 1).reshape([1, n_sample2])
        return abs(np.tile(A, (n_sample2, 1)) + np.tile(B.T, (1, n_sample1)) - 2 * (Y).dot(X.T))

    @staticmethod
    def calc_kernelgram_matrix(*matrix, sigma):
        """Calculate kernel gram matrix itself or between 2 matrices
        example     gram = MathUtils.calc_kernelgram_matrix(X, Y, 0.1)
                    gram = MathUtils.calc_kernelgram_matrix(X, 0.1)

        Args:
            X (TYPE): 2-D matrix
            Y (TYPE): 2-D matrix
            sigma: RBF kernel parameter

        Returns:
            kernel_gram_matrix: 

        """
        if len(matrix) == 1:
            D = MathUtils.calc_distance_matrix(matrix[0])
        else:
            D = MathUtils.calc_l2distance_matrix(matrix[0], matrix[1])
        return np.exp(-0.5 * D / (sigma**2))

class Utils:
    @staticmethod
    def create_label(n_set, n_class):
        tmp = np.arange(0, n_class)
        tmp_label = np.tile(tmp, [n_set, 1])
        label = tmp_label.flatten('F')
        return label