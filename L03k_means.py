# Created on Tue Jun 20 11:14:28 2017
# @author: tom
#
# X = np.array([[1, 2, 3, 4], [1, 1, 1, 1], [1, 2, 1, 2]])
# C = np.array([[3, 3, 3, 3], [1, 1, 1, 1]])

import numpy as np
import matplotlib.pyplot as plt


def log_sum_exp_over_rows(a):
    # This computes log(sum(exp(a), axis=0)) in a numerically stable way
    maxs_small = np.reshape(np.amax(a, axis=0),(1,-1), order="F")
    maxs_big = np.tile(maxs_small, (a.shape[0], 1))
    ret = np.log(np.sum(np.exp(a - maxs_big), axis=0)) + maxs_small
    return ret


def soft_max(Z):
    # MUSEL JSEM POUZIT FUNKCI z assignmentu 3!!!!
    class_normalizer = log_sum_exp_over_rows(Z)
    log_class_prob = Z - np.tile(class_normalizer, (Z.shape[0], 1))
    return np.exp(log_class_prob)


def distance_matrix(X, C, U, norm='L2'):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           C numpy array kxd, matrix of k d-dimensional cluster centres
           U numpy array kxn, matrix of weights
           norm string, defines metrics (possible 'L1', 'L2')
    output: D numpy array kxn, matrix of distances between every observation
            and every center
    uses: np.shape(), np.tile(), np.transpose(), np.sum()
    objective: to find difference between every observation and every
               center in every dimension
    """
    k, n = np.shape(U)
    X = np.tile(X, (k, 1, 1))
    C = np.tile(C, (n, 1, 1))
    C = np.transpose(C, [1, 0, 2])
    C = X - C  # X_knd - C_knd
    if norm == 'L2':
        return np.sum(C ** 2, axis=2)
    elif norm == 'L1':
        return np.sum(np.abs(C), axis=2)
    elif norm == 'M':
        D = []
        for cluster in range(k):
            # I am not sure, if it is ok to use X, as it was used before,
            # but I hope it will free the RAM (original X is probably large)
            X = C[cluster, :, :]
            V = np.cov(X, aweights=U[cluster, :], rowvar=False)
            d = np.shape(V)[0]
            VD = V / (np.linalg.det(V) ** (1 / d))
            VI = np.linalg.inv(VD)
            D.append(np.sum(np.dot(X, VI) * X, axis=1))
        return np.array(D)
    else:
        print('unknown norm, returning zeros!')
        return np.zeros_like(C)


def partition_matrix(D, version='softmax', fuzzyfier=2):
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
    if version == 'softmax':
        D = soft_max(-D)
    elif version == 'fuzzy':
        D = 1 / (D + np.exp(-100))
        D = D / np.sum(D, axis=0, keepdims=True)
    elif version == 'hard':
        indices = np.argmin(D, axis=0)
        D = np.zeros_like(D)
        D[indices, np.arange(np.shape(D)[1])] = 1
    return D ** fuzzyfier


def new_centroids(X, U, k, d):
    """
    input: U numpy array kxn, matrix of weights
           X numpy array nxd, matrix of n d-dimensional observations
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
    uses:
    objective: calculate new centroids
    """
    C = np.zeros((k, d))
    for centroid in range(k):
        U_part = np.tile(U[centroid, :], (d, 1)).T
        C[centroid, :] = (np.sum(U_part * X, axis=0) / np.sum(U_part, axis=0))
    # 1.3 krat pomalejsi verze:
    # X_knd = np.tile(X, (k, 1, 1))
    # U_dkn = np.tile(U, (d, 1, 1))
    # U_knd = np.transpose(U_dkn, [1, 2, 0])
    # C_kn = np.sum(U_knd * X_knd, axis=2) / np.sum(U_knd, axis=2)
    return C


def initialization(X, k, method='random'):
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
    else:
        print('unknown method of initialization, returning zeros!')
        C = np.zeros((k, d))
    return C, U


def k_means(X, k, method='random', norm='L2', version='softmax', fuzzyfier=2,
            visualise=False, iterations=100):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           k positive integer, number of clusters
           method string, defines type of initialization, possible ('random')
           norm string, defines metrics (possible 'L1', 'L2')
           version string, version of making weights (possible 'hard', 'fuzzy',
           'softmax')
           fuzzyfier number, larger or equal one, not too large
           visualise boolean, possibility for 2D data to visualize some steps
           iterations integer, max number of iterations
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
    uses: np.shape(), np.sum(),
          initialization(), distance_matrix(), partition_matrix(),
          new_centroids(), visualisation()
    objective: perform some kind of k-means
    """
    d = np.shape(X)[1]
    J_old = 0
    C, U = initialization(X, k, method)
    for iteration in range(iterations):
        D = distance_matrix(X, C, U, norm)
        U = partition_matrix(D, version, fuzzyfier)
        C = new_centroids(X, U, k, d)
        J_new = np.sum(U * D)
        if abs(J_old - J_new) < 0.01:
            print('no changes! breaking loop.')
            print('iteration: ', iteration)
            print(J_new)
            if visualise:
                visualisation(X, C)
            break
        if iteration % 100 == 0:
            print(J_new)
            if visualise:
                visualisation(X, C)
        J_old = J_new
    return C, U


def visualisation(X, C):
    """
    input: krok int, length of the step
           t numpy array, vector of targets
           y numpy array of the same length as y, vector of predictions
    output: none
    uses: matplotlib.pyplot.*np.c_
    objective: show series of grph devided by the krok
    """
    plt.scatter(X[:,0], X[:,1], color='red', label='DATASET')
    plt.scatter(C[:,0], C[:,1], color='blue', label='CENTRA')
    plt.title('shlukova analyza')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()



# testovani
# X = np.array([[1, 2, 3, 4], [1, 1, 1, 1], [1, 2, 1, 2]])
# C = np.array([[3, 3, 3, 3], [1, 1, 1, 1]])
#k = 3
#X = np.r_[np.random.randn(50, 2), np.random.randn(50, 2) + 3]
#
#
#C, U = k_means(X, k, method='random', norm='M', version='fuzzy', fuzzyfier=2,
#               visualise=True)
#












