# Created on Wed Aug 23 09:43:31 2017
# @author: tom

import gc
import numpy as np

# for fremen


def distance_matrix(X, C, U, structure):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           C numpy array kxd, matrix of k d-dimensional cluster centres
           U numpy array kxn, matrix of weights
           structure list(int, list(floats)), number of non-hypertime
                                              dimensions and list of hypertime
                                              radii
    output: D numpy array kxn, matrix of distances between every observation
            and every center
            COV numpy array kxdxd (?), matrix of cluster covariance
    uses: np.shape(), np.tile(), np.cov(), np.shape(), np.linalg.det(),
          np.linalg.inv(), np.sum(), np.dot(), gc.collect(), np.array()
    objective: to find difference between every observation and every
               center in every dimension
    """
    k, n = np.shape(U)
    D = []
    COV = []
    for cluster in range(k):
        Ci = np.tile(C[cluster, :], (n, 1))
        # hypertime version of X - Ci_nxd
        XC = hypertime_substraction(X, Ci, structure)
        V = np.cov(XC, aweights=U[cluster, :], rowvar=False)
        d = np.shape(V)[0]
        VD = V / (np.linalg.det(V) ** (1 / d))
        VI = np.linalg.inv(VD)
        D.append(np.sum(np.dot(XC, VI) * XC, axis=1))
        COV.append(VI)
        gc.collect()
    return np.array(D), np.array(COV)


def partition_matrix(D, version='fuzzy', fuzzyfier=2):
    """
    input: D numpy array kxn, matrix of distances between every observation
           and every center
           version string, version of making weights (possible 'hard', 'fuzzy',
           'softmax')
           fuzzyfier number, larger or equal one, not too large
    output: U numpy array kxn, matrix of weights
    uses: np.argmin(), np.exp(), np.sum(), np.zeros_like(), np.arange(),
          np.shape(), soft_max()
    objective: to create partition matrix (weights for new centroids
                                           calculation)
    """
    if version == 'fuzzy':
        D = 1 / (D + np.exp(-100))
        D = D / np.sum(D, axis=0, keepdims=True)
    elif version == 'probability':
        U = 1 / (D + np.exp(-100))
        # U[D < 1.5] = 0.67  # zvazit...
        U[D < 1] = 1
        U[D > 16] = 0
        D = np.empty_like(U)
        np.copyto(D, U)
    return D ** fuzzyfier


def new_centroids(X, U, k, d):
    """
    input: U numpy array kxn, matrix of weights
           X numpy array nxd, matrix of n d-dimensional observations
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
    uses: np.zeros(), np.tile(), np.sum()
    objective: calculate new centroids
    """
    C = np.zeros((k, d))
    for centroid in range(k):
        U_part = np.tile(U[centroid, :], (d, 1)).T
        C[centroid, :] = (np.sum(U_part * X, axis=0) / np.sum(U_part, axis=0))
    return C


def initialization(X, k, method='random', C_in=0, U_in=0, structure=[1, []]):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           k positive integer, number of clusters
           method string, defines type of initialization, possible ('random')
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
    uses: np.shape(), np.random.randn(), np.sum(), np.zeros(),
          new_centroids()
    objective: create initial centroids
    """
    if method == 'random':
        n, d = np.shape(X)
        C = X[np.random.choice(np.arange(n), size=k, replace=False), :]
        U = np.zeros((k, n))
        D = distance_matrix(X, C, U, norm='L2')
        U = partition_matrix(D, version='fuzzy', fuzzyfier=2)
    elif method == 'old_C_U':
        C = C_in
        U = U_in
    elif method == 'prev_dim':
        # supposing that the algorith adds only one circle per iteration
        d = np.shape(X)[1]
        C = np.empty(k, d)
        # known part of C
        d_in = np.shape(C_in)[1]
        C[:, : d_in] = C_in
        # unknown part of C lying randomly (R) on the circle with radius r
        R = np.random.rand(k, 1)
        r = structure[1][-1]
        C[:, d_in:] = np.c_[r * np.cos(2*np.pi * R),
                            r * np.sin(2*np.pi * R)]
        U = U_in
    else:
        print('unknown method of initialization, returning zeros!')
        C = np.zeros((k, d))
    return C, U


def k_means(X, k, structure, method='random', version='fuzzy', fuzzyfier=2,
            iterations=100, C_in=0, U_in=0):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           k positive integer, number of clusters
           method string, defines type of initialization, possible ('random',
                                                        'old_C_U', 'prev_dim')
           version string, version of making weights (possible 'fuzzy',
           'probability')
           fuzzyfier number, larger or equal one, not too large
           iterations integer, max number of iterations
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
    C, U = initialization(X, k, method, C_in, U_in)
    for iteration in range(iterations):
        D, COV = distance_matrix(X, C, U, structure)
        U = partition_matrix(D, version, fuzzyfier)
        C = new_centroids(X, U, k, d)
        J_new = np.sum(U * D)
        if abs(J_old - J_new) < 0.01:
            print('no changes! breaking loop.')
            print('iteration: ', iteration)
            print(J_new)
            break
        if iteration % 10 == 0:
            print(J_new)
        J_old = J_new
    densities = np.sum(U, axis=1, keepdims=True) / np.sum(U)
    return C, U, COV, densities


def hypertime_substraction(X, Ci, structure):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           Ci_nxd numpy array nxd, matrix of n d-dimensional cluster centre
                                   copies
           structure list(int, list(floats)), number of non-hypertime
                                              dimensions and list of hypertime
                                              radii
    output: XC numpy array nxWTF, matrix of n WTF-dimensional substractions
    uses:
    objective: to substract C from X in hypertime
    """
    XC = np.empty_like(X)
    # non-hypertime dimensions substraction
    dim = structure[0]
    XC[:, : dim] = X[:, : dim] - Ci[:, : dim]
    # hypertime dimensions substraction
    radii = structure[1]
    for period in range(len(radii)):
        dim = dim + period * 2
        r = radii[period]
        XC[:, dim: dim + 2] = r * np.arccos(np.sum(X[:, dim: dim + 2] *
                                                   Ci[:, dim: dim + 2],
                                                   axis=1, keepdims=True) /
                                            (r ** 2))
    return XC







































































































