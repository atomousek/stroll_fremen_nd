# Created on Thu Jul  6 00:32:42 2017
# @author: tom

import numpy as np


def distance_matrix(X, C, COV):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           C numpy array kxd, matrix of k d-dimensional cluster centres
           COV numpy array kxdxd, matrix of covariance matrices
    output: D numpy array kxn, matrix of distances between every observation
            and every center
    uses: np.shape(), np.tile(), np.transpose(), np.sum(), np.dot(), np.array()
    objective: to find distance based on covariance between every observation
               and every center
    """
    n = np.shape(X)[0]
    k = np.shape(C)[0]
    X = np.tile(X, (k, 1, 1))
    C = np.tile(C, (n, 1, 1))
    C = np.transpose(C, [1, 0, 2])
    C = X - C  # X_knd - C_knd
    D = []
    for cluster in range(k):
        XCp = C[cluster, :, :]
        VI = COV[cluster, :, :]
        D.append(np.sum(np.dot(XCp, VI) * XCp, axis=1))
    return np.array(D)


def partition_matrix(D):
    """
    input: D numpy array kxn, matrix of distances between every observation
           and every center
    output: U numpy array kxn, matrix of weights
    uses: np.exp(), np.sum()
    objective: to create partition matrix
    """
    U = 1 / (D + np.exp(-100))
    U[D < 1] = 1
    return U #/ np.sum(U, axis=0, keepdims=True)


def velocity(U):
    """
    input: U numpy array kxn, matrix of weights
    output: ? numpy array kx1, matrix of sums of weights
    uses: np.sum()
    objective: to find the velocity of generating objects of every distribution
               modeled as a cluster
    """
    total = np.sum(U)
    if total > 0:
        return np.sum(U, axis=1, keepdims=True) / np.sum(U)
    else:
        return np.sum(U, axis=1, keepdims=True)


def density(X, C, COV):

    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           C numpy array kxd, matrix of k d-dimensional cluster centres
           COV numpy array kxdxd, matrix of covariance matrices
    output: ? numpy array kx1, matrix of sums of weights
    uses: distanve_matrix(), partition_matrix(), velocity()
    objective: to find densities of clusters
    """
    D = distance_matrix(X, C, COV)
    U = partition_matrix(D)
    return velocity(U)





