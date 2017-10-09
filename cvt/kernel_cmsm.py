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

        
    def fit(self, data_list, label, n_kgds_dim = None):
        '''
        Apply KPCA and Generate KGDS
        kcmsm.fit(data_list, label)
        ---------
        data_list: Including data matrix. data_list[k] is k-th data matrix. data_list[k] is (n_feature_dimension x n_set) matrix.
        label: This i-th element indicates to which class data_list[i] belongs. Type of label is np.ndarray. Range of label's value is [0, n_class - 1].
        n_kgds_dim: If this value is set, then reference subspaces are projected into n_prjsub_dim-dimension kgds.
        ---------
        '''  
        self.n_kgds_dim = n_kgds_dim
        self.n_dim = data_list[0].shape[0]
        self.n_class = len(np.unique(label))        
        self.X = np.transpose(np.hstack(data_list[:]))
        self.data_list = data_list
        self.label = label

        # calculate kernel matrix
        K = MathUtils.calc_kernelgram_matrix(self.X, self.X, sigma=self.n_sigma)
        
        # E is a matrix, which is used to construct the D. About D, please see Fukui-sensei's TPAMI paper.
        E = np.zeros((K.shape[0], self.n_subdim*self.n_class))

        # 
        i_subdim = np.arange(0, E.shape[1]+ self.n_subdim, self.n_subdim)
        i_data = np.hstack((0, np.cumsum([len(np.where(label==i)[0]) for i in range(self.n_class)])))
        
        sub_basis = []
        
        # KPCA + calculate E
        for c in range(0, self.n_class):
            ## KPCA phase

            ### Extract class kernel matrix from entire kernel matrix.
            Kc = K[i_data[c]:i_data[c+1], i_data[c]:i_data[c+1]]
            [eig_val, eig_vec] = np.linalg.eigh(Kc)

            ### Sort eigenvalues in descending order and extract eigenvector as many as the requested number(n_subdim).            
            eig_id = np.argsort(eig_val)[::-1]
            eig_id = eig_id[0:self.n_subdim]
            eig_val = eig_val[eig_id]
            eig_vec = eig_vec[:,eig_id]

            sub_basis.append(eig_vec / np.sqrt(eig_val))

            ## E is a block diagonal matrix
            E[i_data[c]:i_data[c+1], i_subdim[c]:i_subdim[c+1]] = sub_basis[c]

        # Generate KGDS 
        D = np.dot(np.transpose(E), K)
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
        self.sub_basis = sub_basis
        self.kgds_proj = np.dot(self.gds_basis, np.transpose(E))
 
        self.prj_subspaces = []

        ## Project reference kernel subspace into KGDS
        if ~(n_kgds_dim is None):
            for c in range(self.n_class):
                # temporarily, project basis vectors into KGDS. When calculate similarity, these are orthogonalized.
                self.prj_subspaces.append( np.dot(self.vectors_gds_projection(get_clsdata(self.data_list, self.label, c)), self.sub_basis[c]) )


        

    def vectors_gds_projection(self, data_matrix):
        '''
        Apply KGDS projection for input vectors
        projected_data = kcmsm.vectors_gds_projecton(data_matrix)
        -----------
        data_matrix: 2D-matrix (n_feature_dim x n_data)
        -----------

        Output: projected_data = [self.n_class * self.n_subdim, n_data]
                which is projected data into KPCS + KGDS.

        '''
        data_matrix = np.transpose(data_matrix)
        if data_matrix.ndim == 1:
            data_matrix = data_matrix[np.newaxis, :]

        K = MathUtils.calc_kernelgram_matrix(data_matrix, self.X, sigma=self.n_sigma)

        return np.dot(self.kgds_proj, K)

    def subspaces_gds_projection(self, input_data, input_subspace = None, n_subdim = None):
        '''
         Apply KGDS projection for input subspaces. If input_subspace are not set, firstly apply KPCA to input_data (n_subdim must be set).
         projected_subspaces = subspaces_gds_projection(data, input_subspace = basis).
         projected_subspaces = subspaces_gds_projection(data, n_subdim = n_subdim).
         
         -----------------
         input_data: = [n_feature_dim, n_samples]. This data construct input subspace.
         input_subspace: [n_samples x subspace dim]
         n_subdim: dimension of subspace generated by KPCA. 
         Output:
              projected_subspaces is projected subspace into KGDS. projected_subspaces = [self.n_kgds_dim, subspace dim]
         -----------------

        '''
        if input_subspace is None and n_subdim is None:
            print('Please set \"input_subspace\" or \"n_subdim\"')
            return None
        
        if input_subspace is None:
            ## KPCA phase
            Xin = np.transpose(input_data)
            ### Extract class kernel matrix from entire kernel matrix.
            Kin = MathUtils.calc_kernelgram_matrix(Xin, Xin, sigma=self.n_sigma)
            [eig_val, eig_vec] = np.linalg.eigh(Kin)

            ### Sort eigenvalues in descending order and extract eigenvector as many as the requested number(n_subdim).            
            eig_id = np.argsort(eig_val)[::-1]
            eig_id = eig_id[0:n_subdim]
            eig_val = eig_val[eig_id]
            eig_vec = eig_vec[:,eig_id]

            input_subspace = eig_vec / np.sqrt(eig_val)

        projected = self.vectors_gds_projection(input_data)
        projected = np.dot(projected, input_subspace)
        projected = projected[0:self.n_kgds_dim, :]
        q, _ = np.linalg.qr(projected)
                
        return q
    
    def get_sim(self, input_data, input_subspace = None, n_in_subdim = None):
        '''
        Calculate similarities on self.n_kgds_dim-dimension KGDS 
        If you want to change KGDS dimension, please conduct \" kcmsm.n_kgds_dim = objective_value \".

        *******Usage1:********
        similarities = get_sim(input_data, input_subspace)
        Calculate similarity between each class subspace and a input subspace.
        --------------
        input_data = [n_feature_dim, n_samples].
        input_subspace = [n_samples, n_subspacedim]. The input subspace is generated by apply KPCA to input_data.
        n_in_subdim: This value is not used in Usage1.
        --------------

        *******Usage2:********
        similarities = get_sim(input_data, n_in_subdim = I)
        Firstly, apply KPCA to input_data. Next, Calculate similarity between each class subspace and the generated subspace.
        --------------
        input_data = [n_feature_dim, n_samples].
        n_in_subdim: Input subspace dimension.
        --------------

        Output: similarities = np.ndarray([n_class])
                similarities[c] is similarity between input and c-th class subspace
        '''
        
        if input_subspace is None and n_in_subdim is None:
            print('Please set \"Input subspace basis\" or \"Input subspace dimension\"')
            return []
        
        similarities = np.ndarray((self.n_class))
        for c in range(self.n_class):
            # get c-class training data and reference subspace projected KGDS.
            dict_sub = self.prj_subspaces[c]
            dict_sub = dict_sub[0:self.n_kgds_dim, :]
            dict_sub, _  = np.linalg.qr(dict_sub)
            dict_sub = np.transpose(dict_sub)
            dict_sub = dict_sub[np.newaxis, :, :]
              
            if input_subspace is None:
                ## Apply KPCA to input_Data
                
                # Convert input data to cvtpython form.
                X2 = np.transpose(input_data)
                Kin = MathUtils.calc_kernelgram_matrix(X2, X2, sigma=self.n_sigma)
                [eig_val, eig_vec] = np.linalg.eigh(Kin)

                ### Sort eigenvalues in descending order and extract eigenvector as many as the requested number(n_subdim).            
                eig_id = np.argsort(eig_val)[::-1]
                eig_id = eig_id[0:n_in_subdim]
                eig_val = eig_val[eig_id]
                eig_vec = eig_vec[:,eig_id]
 
                input_subspace = (eig_vec / np.sqrt(eig_val))
                
            prj_sub = self.subspaces_gds_projection(input_data = input_data, input_subspace = input_subspace)
            prj_sub = np.transpose(prj_sub)
            prj_sub = prj_sub[np.newaxis, :, :]

            # calculate canonical angles.
            similarities[c] = np.mean(MathUtils.calc_canonical_angles(dict_sub, prj_sub))
            
        return similarities


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


    @staticmethod
    def calc_canonical_angles(X, Y):
        """Calculate canonical angles between subspaces
        example     similarity = MathUtils.calc_basis_vector(X, Y)

        Args:
            X (TYPE): 3-D matrix (n_set, n_subdim, n_dim)
            Y (TYPE): 3-D matrix (n_set, n_subdim, n_dim)

        Returns:
            C: The similarity of subspaces

        """
        [set_X, subdim_X, dim_X] = size(X)
        if np.ndim(Y) == 4:
            [set_Y1, set_Y2, subdim_Y, dim_Y] = size(Y)
            set_Y = set_Y1 * set_Y2
            Y = np.reshape(Y, (set_Y, subdim_Y, dim_Y))
        elif np.ndim(Y) == 3:
            [set_Y, subdim_Y, dim_Y] = size(Y)
            
        C = np.zeros((min(subdim_X, subdim_Y), set_Y, set_X))
        for i in range(set_X):
            for j in range(set_Y):
                tmp = np.linalg.svd(Y[j, :, :].dot(X[i, :, :].T))
                C[:, j, i] = tmp[1]**2

        return C

class Utils:
    @staticmethod
    def create_label(n_set, n_class):
        tmp = np.arange(0, n_class)
        tmp_label = np.tile(tmp, [n_set, 1])
        label = tmp_label.flatten('F')
        return label