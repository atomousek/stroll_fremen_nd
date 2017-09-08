# Created on Wed Aug 23 09:43:31 2017
# @author: tom

import gc
import numpy as np

# for fremen


def distance_matrix(X, C, U, structure, version):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           C numpy array kxd, matrix of k d-dimensional cluster centres
           U numpy array kxn, matrix of weights
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
    output: D numpy array kxn, matrix of distances between every observation
            and every center
            COV numpy array kxdxd (?), matrix of cluster covariance
    uses: np.shape(), np.tile(), np.cov(), np.shape(), np.linalg.det(),
          np.linalg.inv(), np.sum(), np.dot(), gc.collect(), np.array(),
          test_det()
    objective: to find difference between every observation and every
               center in every dimension
    """
    if version != 'hard':
        k, n = np.shape(U)
        D = []
        COV = []
        for cluster in range(k):
            Ci = np.tile(C[cluster, :], (n, 1))
            # hypertime version of X - Ci_nxd
            XC = hypertime_substraction(X, Ci, structure)
    #        D.append(np.sum(XC ** 2, axis=1))
    #        if np.any(np.isnan(np.cov(XC, aweights=U[cluster, :], rowvar=False))):
    #            print('kovariance je nan, klastr je: ', cluster)
    #            print('toto je U[cluster, :]: ')
    #            print(U[cluster, :])
    #            print('soucet: ', np.sum(U[cluster, :]))
    #            print('maximum: ', np.max(U[cluster, :]))
    #            print('a toto je XC:')
    #            print(XC)
    #            print('nevazena kovariance: ')
    #            print(np.cov(XC, rowvar=False))
    #        if np.any(np.isinf(np.cov(XC, aweights=U[cluster, :], rowvar=False))):
    #            print('kovariance je inf, klastr je: ', cluster)
    #            print('toto je U[cluster, :]: ')
    #            print(U[cluster, :])
    #            print('soucet: ', np.sum(U[cluster, :]))
    #            print('maximum: ', np.max(U[cluster, :]))
    #            print('a toto je XC:')
    #            print(XC)
    #            print('nevazena kovariance: ')
    #            print(np.cov(XC, rowvar=False))
    #            # vznikne, kdyz je soucet a maximum stejna hodnota
    ##            mala_cisla = np.random.rand(*np.shape(U[cluster, :])) * 1e-6
    #            V = np.cov(XC, aweights=U[cluster, :], ddof=0, rowvar=False)
    #            if np.any(np.isinf(V)):
    #                print('ddof=0 nepomohlo')
    #                print('soucet: ', np.sum(U[cluster, :] + mala_cisla))
    #                print('maximum: ', np.max(U[cluster, :] + mala_cisla))
    #                V = np.identity(len(np.cov(XC, rowvar=False)))
    #                print('pouzil jse jednotkovou matici jako kovarianci')
    #                print('vysledek bude k nicemu')
    #            else:
    #                print('prictei nizke hodnoty k U pomohlo')
    #        else:
    #            V = np.cov(XC, aweights=U[cluster, :], rowvar=False)
            V = np.cov(XC, aweights=U[cluster, :], ddof=0, rowvar=False)
            d = np.shape(V)[0]
            determinant = test_det(V, d)
            VD = V / (determinant)
            VI = np.linalg.inv(VD)
            D.append(np.sum(np.abs(np.dot(XC, VI) * XC), axis=1))  # !!!!np.abs(...)** (1 / d)
            COV.append(VI)
            gc.collect()
        D = np.array(D)
        COV = np.array(COV)
        # COV je odsud nikoli seznam kovariancnich matic, ale seznam bodu,
        # k nimz je shluk nejblize. Pak to pouziji na vypocet opravdove COV
    #    indices = np.argmin(D, axis=0)
    #    COV = np.zeros_like(D)
    #    COV[indices, np.arange(np.shape(D)[1])] = 1
        # a ted je to fuzzy prirazeni
    #    COV = 1 / (D + np.exp(-100))
    #    COV = COV / np.sum(COV, axis=0, keepdims=True)
    #    COV = COV ** 2
    else:
        k, n = np.shape(U)
        D = []
        COV = []
        for cluster in range(k):
            Ci = np.tile(C[cluster, :], (n, 1))
            # hypertime version of X - Ci_nxd
            XC = hypertime_substraction(X, Ci, structure)
            D.append(np.sum(XC ** 2, axis=1))
        D = np.array(D)
        COV = np.array(COV)
    return D, COV


def partition_matrix(D, version):
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
        U = 1 / (D + np.exp(-100))
        U = U / np.sum(U, axis=0, keepdims=True)
    elif version == 'probability':
        U = 1 / (D + np.exp(-100))
        # U[D < 1.5] = 0.67  # zvazit...
        # U[D < 1] = 1
        U[D < 1] = 1
        U[D > 4] = 0
        # U = U / np.sum(U, axis=0, keepdims=True)
    elif version == 'model':
        U = 1 / (D + np.exp(-100))
        # U[D < 1.5] = 0.67  # zvazit...
        # U[D < 1] = 1
        U[D < 1] = 1
        # U[D > 4] = np.exp(-100)
        # U[D > 4] = 0
    elif version == 'hard':
        indices = np.argmin(D, axis=0)
        U = np.zeros_like(D)
        U[indices, np.arange(np.shape(D)[1])] = 1
    elif version == 'tom':
        V = 1 / (D + np.exp(-100))
        V = V / np.sum(V, axis=0, keepdims=True)
        indices = np.argmin(D, axis=0)
        W = np.zeros_like(D)
        W[indices, np.arange(np.shape(D)[1])] = 1
        U = V * W
    return U


def new_centroids(X, U, k, d, fuzzyfier):
    """
    input: U numpy array kxn, matrix of weights
           X numpy array nxd, matrix of n d-dimensional observations
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
    uses: np.zeros(), np.tile(), np.sum()
    objective: calculate new centroids
    """
    U = U ** fuzzyfier
    n = np.shape(X)[0]
    C = np.zeros((k, d))
    for centroid in range(k):
        U_part = np.tile(U[centroid, :], (d, 1)).T
#        if np.all(np.sum(U_part, axis=0)) == 0:
#            print('centrum: ', centroid)
#            print('behem vypoctu centroidu jsou vsechny vahy nulove')
#            # U_part = np.random.rand(*np.shape(U_part))
#            U[centroid, :] = np.ones(n)
#            C[centroid, :] = X[np.random.choice(np.arange(n), size=1, replace=False), :]
#        else:
#            C[centroid, :] = (np.sum(U_part * X, axis=0) / np.sum(U_part, axis=0))
#    return C, U ** (1 / fuzzyfier)  # vraceni U je jen kvuli te chybe, pokud se nebude obevovat, muzeme to zrusit
        C[centroid, :] = (np.sum(U_part * X, axis=0) / np.sum(U_part, axis=0))
    return C


def initialization(X, k, method, C_in, U_in, structure, version):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           k positive integer, number of clusters
           method string, defines type of initialization, possible ('random')
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
        D = distance_matrix(X, C, U, structure, version)[0]
        U = partition_matrix(D, version)
    elif method == 'old_C_U':
        C = C_in
        U = U_in
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
        D, COV = distance_matrix(X, C, U, structure, version)
        U = partition_matrix(D, version)  # !!!!
#        C, U = new_centroids(X, U, k, d, fuzzyfier)
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
    densities = np.sum(U, axis=1, keepdims=True) / np.sum(U)
    print('iteration: ', iteration, ' and C:')
    print(list(C))
    print('leaving clustering')
    return C, U, COV, densities


def hypertime_substraction(X, Ci, structure):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           Ci_nxd numpy array nxd, matrix of n d-dimensional cluster centre
                                   copies
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
    output: XC numpy array nxWTF, matrix of n WTF-dimensional substractions
    uses:
    objective: to substract C from X in hypertime
    """
    # non-hypertime dimensions substraction
    XC = X - Ci
#    observations = np.shape(X)[0]
#    ones = np.ones((observations, 1))
#    dim = structure[0]
#    radii = structure[1]
#    XC = np.empty((observations, dim + len(radii)))
#    XC[:, : dim] = X[:, : dim] - Ci[:, : dim]
#    # hypertime dimensions substraction
#    for period in range(len(radii)):
#        r = radii[period]
#        cos = (np.sum(X[:, dim + (period * 2): dim + (period * 2) + 2] *
#                      Ci[:, dim + (period * 2): dim + (period * 2) + 2],
#                      axis=1, keepdims=True) / (r ** 2))
#        cos = np.minimum(np.maximum(cos, ones * -1), ones)
#        XC[:, dim + period: dim + period + 1] = r * np.arccos(cos)
    return XC


def test_det(V, d):
    determinant = np.linalg.det(V)
    if np.linalg.det(V) < 0:
        print('det(V) je zaporny')
        print('norm det +: ', np.linalg.det(V / (np.abs(determinant) ** (1 / d))))
        print('should be one')
        determinant = (np.abs(determinant) ** (1 / d))
    if np.any(np.isnan(V / (determinant ** (1 / d)))):
        print('podil je nan:')
        print('V je divne:')
        print(V)
        determinant = (1e-22) ** (1 / d)
        print('norm det: ', np.linalg.det(V / (determinant ** (1 / d))))
        print('should be one')
    return determinant





































































































