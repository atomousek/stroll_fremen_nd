# Created on Tue Jul 25 15:43:06 2017
# @author: tom


import dataset_io as dio
import clustering as cl
import numpy as np
import gc

# for fremen


def model_fremen(input_coordinates, overall_sum, structure, path, C_old, U_old,
                 k, shape_of_grid):
    """
    input: input_coordinates numpy array, coordinates for model creation
           overall_sum number (np.float64 or np.int64), sum of all measures
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           path string, path to file
           C_old numpy array kxd, centres from last iteration
           U_old numpy array kxn, weights from last iteration
           k positive integer, number of clusters
           shape_of_grid numpy array dx1 int64, number of cells in every
                                                dimension
    output: hist_probs numpy array, 3D histogram of probabilities over grid
            C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
            DODELAT!!!
    uses: model_parameters(), histogram_probs()
    objective: to create grid of probabilities over time-space
               and pass centres and weights to the next clusters initialization
    """
    C, U, COV, densities = model_parameters(path, structure, C_old, U_old, k)
    # toto budu muset nejak rozumne rozsekat (po 10^8 INPUT*k - zabere 4GB RAM)
    hist_probs = histogram_probs(input_coordinates, C, COV, densities,
                                 structure, k, shape_of_grid, overall_sum)
    return hist_probs, C, U, COV, densities


def model_parameters(path, structure, C_old, U_old, k):
    """
    input: path string, path to file
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           C_old numpy array kxd, centres from last iteration
           U_old numpy array nxd, weights from last iteration
           k positive integer, number of clusters
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
            COV numpy array kxdxd, matrix of covariance matrices
            densities numpy array kx1, matrix of number of measurements
                                       belonging to every cluster
    uses: dio.create_X(), cl.k_means(), covariance_matrices()
    objective: to find model parameters
    """
    X = dio.create_X(dio.loading_data(path), structure)
    C, U, COV, densities = cl.k_means(X, k, structure,  # Gustafson–Kessel
                                      method='prev_dim',  # initialization
                                      version='fuzzy',  # objective function
                                      fuzzyfier=2,  # weighting exponent
                                      iterations=1000,  # 1000
                                      C_in=C_old, U_in=U_old)
    COV = covariance_matrices(X, C, U, structure)
    return C, U, COV, densities


def covariance_matrices(X, C, U, structure):
    """
    input: X numpy array nxd, matrix of n d-dimensional observations
           C numpy array kxd, matrix of k d-dimensional cluster centres
           U numpy array kxn, matrix of weights
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
    output: COV numpy array kxdxd, matrix of covariance matrices
    uses: np.shape(), np.tile(), cl.hypertime_substraction(, np.cov(),
          np.linalg.inv(), np.array()
    objective: to recalculate covariance matrices for model without using
               normalisation of cluster volumes
    """
    k, n = np.shape(U)
    COV = []
    for cluster in range(k):
        C_cluster = np.tile(C[cluster, :], (n, 1))
        # XC = X - C_grid  # METRIKA !!!
        XC = cl.hypertime_substraction(X, C_cluster, structure)
        V = np.cov(XC, aweights=U[cluster, :], rowvar=False)
        V = np.linalg.inv(V)
        COV.append(V)
    return np.array(COV)


def histogram_probs(input_coordinates, C, COV, densities, structure, k,
                    shape_of_grid, overall_sum):
    """
    input: input_coordinates numpy array, coordinates for model creation
           C numpy array kxd, matrix of k d-dimensional cluster centres
           COV numpy array kxdxd, matrix of covariance matrices
           densities numpy array kx1, matrix of number of measurements
                                      belonging to every cluster
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           k positive integer, number of clusters
           shape_of_grid numpy array dx1 int64, number of cells in every
                                                dimension
           overall_sum number (np.float64 or np.int64), sum of all measures
    output: hist_probs numpy array, 3D histogram of probabilities over grid
    uses: iter_over_probs(), np.reshape()
    objective: to create grid of probabilities over time-space
    """
    probs = iter_over_probs(input_coordinates, C, COV, densities, structure, k)
    true_probs = probs * overall_sum / np.sum(probs)
    return true_probs.reshape(shape_of_grid)


def iter_over_probs(input_coordinates, C, COV, densities, structure, k):
    """
    input: input_coordinates numpy array, coordinates for model creation
           C numpy array kxd, matrix of k d-dimensional cluster centres
           COV numpy array kxdxd, matrix of covariance matrices
           densities numpy array kx1, matrix of number of measurements
                                      belonging to every cluster
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           k positive integer, number of clusters
    output: probs numpy array len(input_coordinates)x1, probabilities obtained
                                from model in positions of input_coordinates
    uses: np.shape(), np.empty(), probabilities()
    objective: to call probabilities() above smaller parts of input_coordinates
               every part of 10^8 * k lines - takes cca 4GB RAM per call
    """
    number_of_coordinates = np.shape(input_coordinates)[0]
    volume_of_data = number_of_coordinates * k
    number_of_parts = (volume_of_data // (5 * 10 ** 7)) + 1
    length_of_part = number_of_coordinates // (number_of_parts)
    probs = np.empty(number_of_coordinates)
    finish = 0
    for i in range(number_of_parts):
        start = i * length_of_part
        finish = (i + 1) * length_of_part - 1
        part = probabilities(input_coordinates[start: finish, :],
                             C, COV, densities, structure, k)
        probs[start: finish] = part
        gc.collect()
    part = probabilities(input_coordinates[finish:, :],
                         C, COV, densities, structure, k)
    probs[finish:] = part
    return probs


def probabilities(data, C, COV, densities, structure, k):
    """
    input: data numpy array, coordinates for model creation
           C numpy array kxd, matrix of k d-dimensional cluster centres
           COV numpy array kxdxd, matrix of covariance matrices
           densities numpy array kx1, matrix of number of measurements
                                      belonging to every cluster
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           k positive integer, number of clusters
    output: probs numpy array len(data)x1, probabilities obtained
                                           from model in positions of data
    uses: dio.create_X(), np.shape(), gc.collect(), np.tile(),
          cl.hypertime_substraction(), np.sum(), np.dot(), np.array(),
          cl.partition_matrix()
    objective: to generate probabilities based on model above data
    """
    X = dio.create_X(data, structure)
    n = np.shape(X)[0]
    gc.collect()
    D = []
    for cluster in range(k):
        C_cluster = np.tile(C[cluster, :], (n, 1))
        # XC = X - C_cluster  # METRIKA!!!!
        XC = cl.hypertime_substraction(X, C_cluster, structure)
        VI = COV[cluster, :, :]
        D.append(np.sum(np.dot(XC, VI) * XC, axis=1))
        gc.collect()
    D = np.array(D)
    gc.collect()
    D = cl.partition_matrix(D, version='probability', fuzzyfier=1)
    gc.collect()
    D = densities * D
    gc.collect()
    return np.sum(D, axis=0)





#
#import matplotlib.pyplot as plt
#import matplotlib as mpl
#from scipy.misc import toimage
#import k_means as km
#
#
#def psti2(INPUT, C, COV, hustoty_shluku):
#    """
#    input:
#    output:
#    uses:
#    objective:
#    """
#    X = np.c_[(24 / (2*np.pi)) * np.cos(2*np.pi * INPUT[:, 2] / 24),
#              (24 / (2*np.pi)) * np.sin(2*np.pi * INPUT[:, 2] / 24),
#              INPUT[:, 0:2]]
#    k = np.shape(C)[0]
#    n = np.shape(X)[0]
#    gc.collect()
#    D = []
#    for cluster in range(k):
#        C_cluster = np.tile(C[cluster, :], (n, 1))
#        XC = X - C_cluster
#        VI = COV[cluster, :, :]
#        D.append(np.sum(np.dot(XC, VI) * XC, axis=1))
#        gc.collect()
#    D = np.array(D)
#    gc.collect()
#    D = km.partition_matrix(D, version='probability', fuzzyfier=1)
#    gc.collect()
#    D = hustoty_shluku * D
#    gc.collect()
#    return np.sum(D, axis=0)
#
#
#def covariance_matrices2(X, C, U):
#    """
#    input: X numpy array nxd, matrix of n d-dimensional observations
#           C numpy array kxd, matrix of k d-dimensional cluster centres
#           U numpy array kxn, matrix of weights
#    output: COV numpy array kxdxd, matrix of covariance matrices
#    uses: np.shape(), np.tile(), np.transpose(), np.cov(), np.linalg.inv(),
#          np.array()
#    objective: to find difference between, structure every observation and every
#               center in every dimension
#    """
#    k, n = np.shape(U)
#    COV = []
#    for cluster in range(k):
#        C_grid = np.tile(C[cluster, :], (n, 1))
#        XC = X - C_grid
#        V = np.cov(XC, aweights=U[cluster, :], rowvar=False)
#        V = np.linalg.inv(V)
#        COV.append(V)
#    return np.array(COV)
#
#
#def model_0(input_coordinates, training_data, k=20,
#            data_location='/home/tom/projects/atomousek/stroll_fremen_nd/' +
#            'priklad.txt'):
#    """
#    input:
#    output:
#    uses:
#    objective:
#    """
#    X = dio.loading_priklad_natural(file_location=data_location)
#    C, U, COV, densities = km.k_means(X, k,  # Gustafson–Kessel Algorithm
#                                      method='random',  # initialization
#                                      norm='M',  # metrics (Mahalanobis)
#                                      version='fuzzy',  # objective function
#                                      fuzzyfier=2,  # weighting exponent
#                                      visualise=False, iterations=1000)
#    COV = covariance_matrices2(X, C, U)
#    # toto budu muset nejak rozumne rozsekat (po 10^8 INPUT*k - zabere 4GB RAM)
#    # from this now on the "density.py" is obsolete
#    number_of_coordinates = np.shape(input_coordinates)[0]
#    volume_of_data = number_of_coordinates * k
#    number_of_parts = (volume_of_data // (10 ** 8)) + 1
#    length_of_part = number_of_coordinates // (number_of_parts)
#    probabilities = np.empty(number_of_coordinates)
#    finish = 0
#    for i in range(number_of_parts):
#        start = i * length_of_part
#        finish = (i + 1) * length_of_part - 1
#        part = psti2(input_coordinates[start: finish, :],
#                     C, COV, densities)
#        probabilities[start: finish] = part
#        gc.collect()
#    part = psti2(input_coordinates[finish:, :],
#                 C, COV, densities)
#    probabilities[finish:] = part
#    # np.sum(training_data) nahradit len(X) a training data neloadovat !!!
#    true_probabilities = probabilities *\
#        np.sum(training_data) / np.sum(probabilities)
#    return true_probabilities
#
#
#def model_visualisation(true_probabilities, training_data, shape_of_grid):
#    """
#    input:
#    output:
#    uses:
#    objective:
#    """
#    H_probs = true_probabilities.reshape(shape_of_grid)
#    H_train = training_data.reshape(shape_of_grid)
#    random_values = np.random.rand(*shape_of_grid)
#    H_test = (random_values < H_probs) * 1
#    # build pictures and save them
#    fig = plt.figure(dpi=400)
#    for i in range(shape_of_grid[2]):
#        # training data
#        plt.subplot(221)
#        cmap = mpl.colors.ListedColormap(['white', 'blue', 'red'])
#        bounds = [-0.5, 0.5, 1.5, 3000]
#        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#        img = plt.imshow(H_train[:, :, i], interpolation='nearest',
#                         cmap=cmap, norm=norm)
#        plt.xticks([])
#        plt.yticks([])
#        # testing data
#        plt.subplot(222)
#        cmap = mpl.colors.ListedColormap(['white', 'blue', 'red'])
#        bounds=[-0.5, 0.5, 1.5, 3000]
#        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#        img = plt.imshow(H_test[:, :, i],interpolation='nearest',
#                            cmap = cmap,norm=norm)
#        plt.xticks([])
#        plt.yticks([])
#        # model
#        plt.subplot(212)
#        cmap = mpl.colors.ListedColormap(['black', 'blue', 'purple', 'red',
#                                          'orange', 'pink', 'yellow', 'white'])
#        bounds=[-0.5, 0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 1]
#        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#        img = plt.imshow(H_probs[:, :, i],interpolation='nearest',
#                            cmap = cmap,norm=norm)
#        plt.colorbar(img, cmap=cmap,
#                     norm=norm, boundaries=bounds, 
#                     ticks=[0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 1],
#                     fraction=0.046, pad=0.01)
#        plt.xticks([])
#        plt.yticks([])
#        # all together
#        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
#        # name the file
#        name = str(i)
#        if len(name.split('.')[0]) == 1:
#            name = '0' + name
#        path = '/home/tom/projects/atomousek/stroll_fremen_nd/output/' +\
#               'images/' + name + '.png'
#        fig.canvas.draw()
#        # really do not understand :) coppied from somewhere
#        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#        plt.clf()
#        # save
#        toimage(data).save(path)
#    plt.close(fig)

