# Created on Wed Aug 23 09:43:31 2017
# @author: tom

import numpy as np


def distance_matrix(X, C, U):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           C numpy array kxd, matrix of k d-dimensional cluster centres
           U numpy array kxn, matrix of weights
    output: D numpy array kxn, matrix of distances between every observation
            and every center
    uses: np.shape(), np.tile(), np.array(), np.abs()
    objective: to find difference between every observation and every
               center in every dimension
    """
    k, n = np.shape(U)
    D = []
    for cluster in range(k):
        Ci = np.tile(C[cluster, :], (n, 1))
        XC = X - Ci
        # L1 metrics
        D.append(np.sum(np.abs(XC), axis=1))
    D = np.array(D)
    return D


def partition_matrix(D, version):
    """
    input: D numpy array kxn, matrix of distances between every observation
           and every center
           version string, version of making weights (possible 'hard', 'fuzzy',
           'model')
    output: U numpy array kxn, matrix of weights
    uses: np.argmin(), np.sum(), np.zeros_like(), np.arange(),
          np.shape()
    objective: to create partition matrix (weights for new centroids
                                           calculation)
    """
    if version == 'fuzzy':
        U = 1 / (D + np.exp(-100))
        U = U / np.sum(U, axis=0, keepdims=True)
    elif version == 'model':
        U = 1 / (D + np.exp(-100))
        U[D < 1] = 1
    elif version == 'hard':
        indices = np.argmin(D, axis=0)
        U = np.zeros_like(D)
        U[indices, np.arange(np.shape(D)[1])] = 1
    return U


def new_centroids(X, U, k, d, fuzzyfier):
    """
    input: U numpy array kxn, matrix of weights
           X numpy array nxd, matrix of n d-dimensional observations
           k positive integer, number of clusters
           d positive integer, number of dimensions
           fuzzyfier number, larger or equal one, not too large, usually 2 or 1
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
    uses: np.zeros(), np.tile(), np.sum()
    objective: calculate new centroids
    """
    U = U ** fuzzyfier
    C = np.zeros((k, d))
    for centroid in range(k):
        U_part = np.tile(U[centroid, :], (d, 1)).T
        C[centroid, :] = (np.sum(U_part * X, axis=0) / np.sum(U_part, axis=0))
    return C


def initialization(X, k, method, C_in, U_in, structure, version):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           k positive integer, number of clusters
           method string, defines type of initialization, (possible 'random',
                                                           'prev_dim')
           C_in numpy array kxd, matrix of k d-dimensional cluster centres
                                 from the last iteration
           U_in numpy array kxn, matrix of weights from the last iteration
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
    uses: np.shape(), np.random.randn(), np.sum(), np.zeros()
            DODELAT!!!
    objective: create initial centroids
    """
    if method == 'random':
        n, d = np.shape(X)
        C = X[np.random.choice(np.arange(n), size=k, replace=False), :]
        U = np.random.rand(k, n)
        D = distance_matrix(X, C, U)
        U = partition_matrix(D, version)
    elif method == 'prev_dim':
        # supposing that the algorith adds only one circle per iteration
        d = np.shape(X)[1]
        C = np.empty((k, d))
        # known part of C
        d_in = np.shape(C_in)[1]
        C[:, : d_in] = C_in
        # unknown part of C lying randomly (R) on the circle with radius r
        if d_in + 2 == d:
            R = np.random.rand(k, 1)
            r = structure[1][-1]
            C[:, d_in:] = np.c_[r * np.cos(2*np.pi * R),
                                r * np.sin(2*np.pi * R)]
        U = U_in
    else:
        print('unknown method of initialization, returning zeros!')
        C = np.zeros((k, d))
    return C, U


def k_means(X, k, structure, method, version, fuzzyfier,
            iterations, C_in, U_in):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           k positive integer, number of clusters
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           method string, defines type of initialization, possible ('random',
                                                        'old_C_U', 'prev_dim')
           version string, version of making weights (possible 'fuzzy',
           'probability')
           fuzzyfier number, larger or equal one, not too large
           iterations integer, max number of iterations
           DODELAT!!!
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
            COV numpy array kxdxd, matrix of covariance matrices
            densities numpy array kx1, matrix of number of
                    measurements belonging to every cluster
    uses: np.shape(), np.sum(),
          initialization(), distance_matrix(), partition_matrix(),
          new_centroids(), visualisation()
    objective: perform some kind of k-means
    """
    d = np.shape(X)[1]
    J_old = 0
    C, U = initialization(X, k, method, C_in, U_in, structure, version)
    for iteration in range(iterations):
        D = distance_matrix(X, C, U)
        U = partition_matrix(D, version)
        C = new_centroids(X, U, k, d, fuzzyfier)
        J_new = np.sum(U * D)
        if abs(J_old - J_new) < 0.01:
            print('no changes! breaking loop.')
            print('iteration: ', iteration)
            print(J_new)
            break
        if iteration % 10 == 0:
            print(J_new)
        J_old = J_new
    densities = np.sum(U, axis=1, keepdims=True)
    print('iteration: ', iteration, ' and C:')
    print(list(C))
    print('and densities: ')
    print(densities)
    print('leaving clustering')
    return C, U, densities
